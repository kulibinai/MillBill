"""Thin client for the vLLM OpenAI-compatible endpoint with tracing and token accounting."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import SETTINGS, Settings


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls

    def as_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total, "calls": self.calls}


@dataclass
class LLMResponse:
    content: str
    reasoning: str | None
    tool_calls: list[dict]  # [{"id":..., "name":..., "arguments": dict}]
    raw: Any
    usage: Usage
    finish_reason: str = "stop"

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class Trace:
    """Execution trace (tool calls, arguments, results, reasoning, token usage), as in the paper's per-case logs."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def log(self, kind: str, **data: Any) -> None:
        with self._lock:
            self.events.append({"t": time.time(), "kind": kind, **data})

    def dump(self, path) -> None:
        with open(path, "w") as f:
            for e in self.events:
                f.write(json.dumps(e, default=str) + "\n")


class LLM:
    def __init__(self, settings: Settings = SETTINGS, trace: Trace | None = None) -> None:
        self.s = settings
        try:  # openai>=3 ships httpx2; older SDKs use httpx. Local tunnel endpoint: never route via system/env proxies.
            import httpx2 as httpx
        except ImportError:  # pragma: no cover
            import httpx
        self.client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=settings.timeout, max_retries=2,
                             http_client=httpx.Client(trust_env=False, timeout=settings.timeout))
        self.trace = trace or Trace()
        self.usage = Usage()
        self._lock = threading.Lock()
        self._local = threading.local()  # per-thread usage of the last json_call (parallel sub-agents run concurrently)

    @property
    def last_usage(self) -> Usage:
        return getattr(self._local, "usage", None) or Usage()

    @last_usage.setter
    def last_usage(self, u: Usage) -> None:
        self._local.usage = u

    # ------------------------------------------------------------------ core
    def chat(self, messages: list[dict], tools: list[dict] | None = None, *, agent: str = "",
             temperature: float | None = None, thinking: bool | None = None, tool_choice: str | None = None,
             max_tokens: int | None = None, response_json: bool = False) -> LLMResponse:
        thinking = self.s.thinking if thinking is None else thinking
        kwargs: dict[str, Any] = dict(
            model=self.s.model, messages=messages,
            temperature=self.s.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.s.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        if not thinking:  # Qwen non-thinking recommendation: top_p 0.8, presence 1.5; at T=0 mostly irrelevant
            kwargs["top_p"] = 0.8
        if tools:
            kwargs["tools"] = tools
            # Existing vLLM without --enable-auto-tool-choice rejects the SDK default "auto".
            # "none" still puts tools in the prompt; Qwen emits XML that _extract_text_tool_calls parses.
            choice = tool_choice or os.environ.get("CAM_TOOL_CHOICE")
            if choice:
                kwargs["tool_choice"] = choice
        if response_json and not tools:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ != "APIConnectionError" or not self._restart_tunnel():
                raise
            resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        content = msg.content or ""
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = (getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                     or extra.get("reasoning_content") or extra.get("reasoning"))
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        if not calls and tools:  # tolerate models that emit the XML/JSON call as plain text
            calls = _extract_text_tool_calls(content)
        u = Usage(resp.usage.prompt_tokens if resp.usage else 0, resp.usage.completion_tokens if resp.usage else 0, 1)
        with self._lock:
            self.usage.add(u)
        finish = resp.choices[0].finish_reason or "stop"
        self.trace.log("llm", agent=agent, seconds=round(time.time() - t0, 2), usage=u.as_dict(), finish_reason=finish,
                       tool_calls=calls, content=content[:4000], reasoning=(reasoning or "")[:2000])
        return LLMResponse(content, reasoning, calls, resp, u, finish)

    def _restart_tunnel(self) -> bool:
        """The vLLM endpoint is reached through an SSH tunnel that drops when the laptop sleeps: bring it back once."""
        import subprocess
        from .config import ROOT_DIR
        script = ROOT_DIR / "tunnel.sh"
        self._tunnel_retries = getattr(self, "_tunnel_retries", 0) + 1
        # tunnel.sh is not executable in git (mode 100644): without this check a dropped server raised PermissionError
        if "127.0.0.1" not in self.s.base_url or not os.access(script, os.X_OK) or self._tunnel_retries > 5:
            return False
        subprocess.run([str(script), "stop"], capture_output=True, timeout=30)
        r = subprocess.run([str(script), "start"], capture_output=True, text=True, timeout=60)
        self.trace.log("tunnel", restarted=r.returncode == 0)
        return r.returncode == 0

    def assistant_message(self, r: LLMResponse) -> dict:
        content = r.content or ""
        if r.tool_calls and r.tool_calls[0]["id"].startswith("text-"):
            # calls parsed from the XML text: keep them only in tool_calls, otherwise the chat template renders every
            # call twice and the model tends to re-issue the whole batch on the next turn
            cut = min((i for i in (content.find("<tool_call>"), content.find("<function=")) if i >= 0), default=len(content))
            content = content[:cut].rstrip()
        m: dict[str, Any] = {"role": "assistant", "content": content}
        if r.tool_calls:
            m["tool_calls"] = [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in r.tool_calls]
        return m

    # --------------------------------------------------------------- helpers
    def json_call(self, system: str, user: str, *, agent: str = "", temperature: float | None = None,
                  thinking: bool | None = None, retries: int = 2) -> dict:
        """One-shot structured call. Returns a parsed JSON object (empty dict on persistent failure)."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        max_tokens = self.s.max_tokens
        self.last_usage = Usage()
        for attempt in range(retries + 1):
            r = self.chat(messages, agent=agent, temperature=temperature, thinking=thinking, response_json=True, max_tokens=max_tokens)
            self.last_usage.add(r.usage)
            obj = None if r.truncated else parse_json(r.content)
            if isinstance(obj, dict):
                return obj
            messages.append({"role": "assistant", "content": r.content})
            if r.truncated:
                max_tokens = min(max_tokens * 2, 24000)
                messages.append({"role": "user", "content": TRUNCATED_HINT})
            else:
                messages.append({"role": "user", "content": "Your reply was not a single valid JSON object. Reply with only the JSON object."})
        return {}


TRUNCATED_HINT = ("Your answer was cut off at the output token limit. Reply again with ONLY the JSON object and make it "
                  "compact: merge identical entries across features into one entry with a feature_id list, keep messages "
                  "short, and drop purely informational items.")


_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def parse_json(text: str) -> Any:
    """Best-effort extraction of the first JSON object in a model reply."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None  # unbalanced: the object was truncated


_XML_CALL = re.compile(r"<function=([\w\-.]+)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w\-.]+)>(.*?)</parameter>", re.S)


def _extract_text_tool_calls(content: str) -> list[dict]:
    calls = []
    for i, m in enumerate(_XML_CALL.finditer(content or "")):
        args = {}
        for pm in _XML_PARAM.finditer(m.group(2)):
            v = pm.group(2).strip()
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            args[pm.group(1)] = v
        calls.append({"id": f"text-{i}", "name": m.group(1), "arguments": args})
    return calls
