"""Guardrails applied around every LLM call and every outbound message.

Input side : prompt-injection screening of *untrusted* text (event feeds, retrieved
             docs, rider free text) + PII redaction before text reaches a model.
Output side: schema validation (pydantic, in llm/client.py) + content policy checks
             for driver/rider messages derived from the comms style guides.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now", r"new instructions:", r"act as (an? )?(admin|developer|system)",
    r"set (the )?surge to", r"approve (this|the) plan", r"bypass (the )?(policy|compliance|approval)",
    r"</?(system|assistant|tool)>",
]
_inj = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

PII_PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "PHONE": r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
    "CARD": r"\b(?:\d[ -]?){13,16}\b",
    "PLATE": r"\bT\d{6}C\b",  # NYC TLC FHV plate format
}

UNSAFE_DRIVER_PHRASES = [
    r"\bhurry\b", r"\brace\b", r"\bspeed(ing)? up\b", r"\bfast as (you )?can\b", r"\bnow or never\b",
    r"\bwait (at|on) the curb\b", r"\bbus lane\b", r"\bguarantee(d)? (earnings|income)\b", r"\bdouble[- ]park",
]


@dataclass
class ScreenResult:
    text: str
    flagged: bool
    reasons: list[str] = field(default_factory=list)


def screen_untrusted(text: str) -> ScreenResult:
    """Neutralise instruction-like content in data that came from outside (feeds, docs, users)."""
    reasons = [m.group(0) for m in _inj.finditer(text)]
    cleaned = _inj.sub("[removed: instruction-like content]", text)
    return ScreenResult(cleaned, bool(reasons), reasons)


def wrap_untrusted(label: str, text: str) -> str:
    """Delimit untrusted context so the model treats it as data, never as instructions."""
    return f"<untrusted_data source=\"{label}\">\n{screen_untrusted(text).text}\n</untrusted_data>"


def redact_pii(text: str) -> str:
    for tag, pat in PII_PATTERNS.items():
        text = re.sub(pat, f"[{tag}]", text)
    return text


def check_driver_message(msg: str, committed_bonus: float | None = None, max_len: int = 140) -> list[str]:
    issues = []
    if len(msg) > max_len:
        issues.append(f"push text {len(msg)} chars > {max_len}")
    for pat in UNSAFE_DRIVER_PHRASES:
        if re.search(pat, msg, re.IGNORECASE):
            issues.append(f"unsafe/forbidden phrase: {pat}")
    amounts = [float(a) for a in re.findall(r"\$(\d+(?:\.\d+)?)", msg)]
    if committed_bonus is not None and any(a > committed_bonus + 1e-6 for a in amounts):
        issues.append(f"mentions amount above committed bonus ${committed_bonus}")
    return issues


def check_rider_message(msg: str, surge: float) -> list[str]:
    issues = []
    if surge > 1.05 and not re.search(r"higher|surge|busier|increased", msg, re.IGNORECASE):
        issues.append("must disclose that prices are higher than usual")
    if not re.search(r"\d+\s*(?:-|to|–)?\s*\d*\s*min", msg):
        issues.append("must include an estimated wait")
    return issues
