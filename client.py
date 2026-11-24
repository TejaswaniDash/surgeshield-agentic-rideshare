"""Provider-agnostic LLM client with structured (pydantic) outputs, self-repair and accounting.

Providers
  anthropic : Claude via the Anthropic SDK (ANTHROPIC_API_KEY)
  openai    : any OpenAI-compatible /chat/completions server (Ollama, vLLM, OpenAI) via httpx
  mock      : deterministic, offline - builds schema-valid outputs from the call's context.
              Lets the whole agentic system run in CI and on a laptop with no keys.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from surgeshield.config import get_settings
from surgeshield.observability.tracing import incr, span

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    text: str
    parsed: Any = None
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    repairs: int = 0
    meta: dict = field(default_factory=dict)


def _extract_json(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start >= 0 and end > start else text


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        cfg = get_settings().llm
        self.cfg = cfg
        self.provider = (provider or os.environ.get("LLM_PROVIDER") or cfg.provider).lower()
        if self.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            log.warning("ANTHROPIC_API_KEY not set - falling back to mock LLM")
            self.provider = "mock"
        self.model = model or (cfg.openai_model if self.provider == "openai" else cfg.model)
        self._client = None

    # ---------------------------------------------------------------- providers
    def _call(self, system: str, prompt: str, json_mode: bool, task: str, context: dict) -> tuple[str, int, int]:
        if self.provider == "mock":
            from surgeshield.llm.mock import mock_response
            out = mock_response(task, context)
            return out, len(prompt) // 4, len(out) // 4
        if self.provider == "anthropic":
            import anthropic
            self._client = self._client or anthropic.Anthropic()
            resp = self._client.messages.create(
                model=self.model, max_tokens=self.cfg.max_tokens, temperature=self.cfg.temperature,
                system=system, messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return text, resp.usage.input_tokens, resp.usage.output_tokens
        if self.provider == "openai":
            import httpx
            body: dict[str, Any] = {"model": self.model, "temperature": self.cfg.temperature,
                                    "max_tokens": self.cfg.max_tokens,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": prompt}]}
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}"}
            r = httpx.post(f"{self.cfg.openai_base_url}/chat/completions", json=body, headers=headers, timeout=120)
            r.raise_for_status()
            j = r.json()
            u = j.get("usage", {})
            return j["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        raise ValueError(f"unknown provider {self.provider}")

    # ---------------------------------------------------------------- public API
    def generate(self, task: str, system: str, prompt: str, schema: type[T] | None = None,
                 context: dict | None = None) -> LLMResult:
        """Generate text, or a validated `schema` instance with up to N self-repair rounds."""
        context = context or {}
        if schema is not None:
            prompt = (f"{prompt}\n\nRespond with ONLY a JSON object that validates against this JSON Schema:\n"
                      f"{json.dumps(schema.model_json_schema())}")
        res = LLMResult(text="", provider=self.provider, model=self.model)
        with span("llm.generate", task=task, provider=self.provider, model=self.model) as sp:
            t0 = time.time()
            attempt_prompt = prompt
            for attempt in range(self.cfg.max_repair_attempts + 1):
                text, i_tok, o_tok = self._call(system, attempt_prompt, schema is not None, task, context)
                res.text, res.input_tokens, res.output_tokens = text, res.input_tokens + i_tok, res.output_tokens + o_tok
                if schema is None:
                    break
                try:
                    res.parsed = schema.model_validate_json(_extract_json(text))
                    break
                except (ValidationError, ValueError) as e:
                    res.repairs += 1
                    incr("llm.repairs")
                    attempt_prompt = (f"{prompt}\n\nYour previous answer was invalid:\n{text[:2000]}\n"
                                      f"Validation error: {str(e)[:1500]}\nReturn corrected JSON only.")
            if schema is not None and res.parsed is None:
                raise ValueError(f"LLM failed to produce valid {schema.__name__} after repairs")
            res.latency_ms = round((time.time() - t0) * 1000, 1)
            sp["attrs"].update(input_tokens=res.input_tokens, output_tokens=res.output_tokens, repairs=res.repairs)
        incr("llm.calls")
        incr("llm.input_tokens", res.input_tokens)
        incr("llm.output_tokens", res.output_tokens)
        return res


_default: LLMClient | None = None


def get_llm() -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient()
    return _default
