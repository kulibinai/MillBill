"""Tool registry: bounded, typed tools exposed to ReAct agents via OpenAI function calling."""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict = field(default_factory=dict)  # JSON schema "properties"
    required: list[str] = field(default_factory=list)
    source: str = ""  # provenance label attached to every result

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.parameters, "required": self.required}}}

    def __call__(self, **kwargs: Any) -> Any:
        sig = inspect.signature(self.fn)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            accepted = kwargs
        return self.fn(**accepted)


def tool_result_text(result: Any, limit: int = 6000) -> str:
    """Serialize a tool result for the model, truncating very long payloads."""
    if isinstance(result, str):
        text = result
    else:
        text = json.dumps(result, ensure_ascii=False, default=str, indent=None)
    if len(text) > limit:
        text = text[:limit] + f"... [truncated {len(text) - limit} chars]"
    return text


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.add(t)

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def subset(self, names: list[str]) -> "ToolRegistry":
        return ToolRegistry([self._tools[n] for n in names if n in self._tools])

    def __len__(self) -> int:
        return len(self._tools)
