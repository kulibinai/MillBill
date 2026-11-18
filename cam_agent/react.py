"""ReAct agents (reason -> act -> observe -> repeat -> synthesize).

Two configurations, mirroring Sec. 3.4 of Design-to-Plan:
  * SequentialReAct : one reasoning agent, full tool registry, iterates until it answers.
  * ParallelReAct   : several sub-agents with restricted tool scopes run concurrently, each doing a
                      restricted ReAct loop; a coordinator merges their outputs.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .llm import LLM, TRUNCATED_HINT, Usage, parse_json
from .tools import ToolRegistry, tool_result_text


@dataclass
class AgentResult:
    agent: str
    output: dict
    tool_calls: list[dict] = field(default_factory=list)  # {"name","arguments","result_preview"}
    iterations: int = 0
    tokens: int = 0
    seconds: float = 0.0
    success: bool = True
    error: str | None = None
    sub_results: list["AgentResult"] = field(default_factory=list)

    @property
    def tools_called(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]

    def as_dict(self) -> dict:
        return {"agent": self.agent, "success": self.success, "error": self.error, "iterations": self.iterations,
                "tokens": self.tokens, "seconds": round(self.seconds, 2), "tools_called": self.tools_called,
                "output": self.output, "sub_results": [s.as_dict() for s in self.sub_results]}


FINAL_INSTRUCTION = (
    "When you have enough evidence, stop calling tools and reply with ONLY a single JSON object "
    "(no prose, no markdown fence) that follows the OUTPUT SCHEMA in your instructions."
)


class SequentialReAct:
    def __init__(self, name: str, system_prompt: str, tools: ToolRegistry, llm: LLM, *, max_iterations: int = 8,
                 guard: Callable[[str, dict, list[str]], str | None] | None = None,
                 temperature: float | None = None, thinking: bool | None = None) -> None:
        self.name, self.system_prompt, self.tools, self.llm = name, system_prompt, tools, llm
        self.max_iterations = max_iterations
        self.guard = guard  # (tool_name, args, tools_called_so_far) -> error text if the call is not allowed
        self.temperature, self.thinking = temperature, thinking

    def run(self, user_input: str) -> AgentResult:
        t0 = time.time()
        res = AgentResult(agent=self.name, output={})
        messages = [{"role": "system", "content": self.system_prompt + "\n\n" + FINAL_INSTRUCTION},
                    {"role": "user", "content": user_input}]
        usage = Usage()
        schemas = self.tools.schemas()
        content = ""
        seen: dict = {}
        max_tokens = self.llm.s.max_tokens
        r = None
        for it in range(1, self.max_iterations + 2):
            res.iterations = it
            last = it > self.max_iterations
            r = self.llm.chat(messages, tools=None if last or not schemas else schemas, agent=self.name,
                              temperature=self.temperature, thinking=self.thinking, max_tokens=max_tokens)
            usage.add(r.usage)
            content = r.content
            if r.truncated and not r.tool_calls:
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": TRUNCATED_HINT})
                max_tokens = min(max_tokens * 2, 24000)
                continue
            if not r.tool_calls or last:
                break
            messages.append(self.llm.assistant_message(r))
            repeated = 0
            for call in r.tool_calls:
                key = (call["name"], json.dumps(call["arguments"], sort_keys=True, default=str))
                if key in seen:  # the same call again: return the cached result and push towards the answer
                    out = {"note": "identical call already answered earlier in this conversation; use that result and answer", "result": seen[key]}
                    repeated += 1
                else:
                    out = self._execute(call["name"], call["arguments"], res.tools_called)
                    seen[key] = out
                res.tool_calls.append({"name": call["name"], "arguments": call["arguments"],
                                       "result_preview": tool_result_text(out, 300)})
                self.llm.trace.log("tool", agent=self.name, name=call["name"], arguments=call["arguments"],
                                   result=tool_result_text(out, 2000))
                messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                                 "content": tool_result_text(out, 3500)})
            if it == self.max_iterations or (repeated and repeated == len(r.tool_calls)):
                messages.append({"role": "user", "content": ("Iteration budget reached. " if it == self.max_iterations else
                                                             "You are repeating calls. ") + FINAL_INSTRUCTION})
                if repeated:
                    self.max_iterations = it  # next turn is the final one (no tools)
        obj = parse_json(content)
        if not isinstance(obj, dict):
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "That was not a valid JSON object. " + FINAL_INSTRUCTION})
            r = self.llm.chat(messages, agent=self.name, temperature=self.temperature, thinking=self.thinking, response_json=True, max_tokens=max_tokens)
            usage.add(r.usage)
            obj = parse_json(r.content)
        if isinstance(obj, dict):
            res.output = obj
        else:
            res.success, res.error = False, "no structured output"
            res.output = {"raw": content}
        res.tokens = usage.total
        res.seconds = time.time() - t0
        return res

    def _execute(self, name: str, args: dict, called: list[str]) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool '{name}'. Available: {self.tools.names()}"}
        if self.guard:
            blocked = self.guard(name, args, called)
            if blocked:
                return {"error": blocked}
        try:
            out = tool(**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}", "expected": list(tool.parameters)}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
        if isinstance(out, dict) and tool.source and "source" not in out:
            out = {"source": tool.source, **out}
        return out


class ParallelReAct:
    """Sub-agents run concurrently; a coordinator merges (Fig. 6b / Appendix A of the paper)."""

    def __init__(self, name: str, sub_agents: list[SequentialReAct], coordinator_prompt: str, llm: LLM, *,
                 temperature: float | None = None, thinking: bool | None = None,
                 merge: Callable[[list[AgentResult]], dict] | None = None) -> None:
        self.name, self.sub_agents, self.coordinator_prompt, self.llm = name, sub_agents, coordinator_prompt, llm
        self.temperature, self.thinking = temperature, thinking
        # merge: deterministic merge of the sub-agent outputs. When given, the coordinator only reviews the merged
        # result and returns the keys it adds (e.g. summary, confidence) instead of re-generating everything:
        # output tokens dominate latency, so re-emitting all rules cost minutes.
        self.merge = merge

    def run(self, user_input: str) -> AgentResult:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(self.sub_agents)) as ex:
            subs = list(ex.map(lambda a: a.run(user_input), self.sub_agents))
        failures = [s.agent for s in subs if not s.success]
        if self.merge:
            merged = self.merge(subs)
            review = {"feature_description": user_input, "merged_result": merged, "sub_agent_failures": failures}
            extra = self.llm.json_call(self.coordinator_prompt, json.dumps(review, ensure_ascii=False, default=str),
                                       agent=self.name + ".coordinator", temperature=self.temperature, thinking=self.thinking)
            obj = {**merged, **{k: v for k, v in extra.items() if k in ("summary", "confidence")}} if merged else extra
        else:
            merged_input = {"feature_description": user_input,
                            "sub_agent_results": {s.agent: s.output for s in subs},
                            "sub_agent_failures": failures}
            obj = self.llm.json_call(self.coordinator_prompt, json.dumps(merged_input, ensure_ascii=False, default=str),
                                     agent=self.name + ".coordinator", temperature=self.temperature, thinking=self.thinking)
        res = AgentResult(agent=self.name, output=obj, sub_results=subs, iterations=1,
                          success=bool(obj), error=None if obj else "coordinator produced no JSON")
        for s in subs:
            res.tool_calls.extend(s.tool_calls)
        res.tokens = sum(s.tokens for s in subs) + self.llm.last_usage.total
        res.seconds = time.time() - t0
        return res
