"""Zen (opencode.ai) free-model teachers. Key from ~/.env (zen=sk-...) or $ZEN_API_KEY.
Never logs key, never commits it. Provenance license: unknown-do-not-redistribute."""
from __future__ import annotations
import os, re, uuid

BASE = "https://opencode.ai/zen/v1"
DEFAULTS = [m.strip() for m in os.environ.get(
    "FORGE_ZEN_MODELS", "mimo-v2.5-free").split(",") if m.strip()]

def _key() -> str:
    if os.environ.get("ZEN_API_KEY"): return os.environ["ZEN_API_KEY"]
    try:
        for line in open(os.path.expanduser("~/.env")):
            m = re.match(r"\s*zen\s*=\s*(sk-\S+)", line)
            if m: return m.group(1).strip()
    except Exception: pass
    return ""

def ask(prompt: str, model: str = "", system: str = "", timeout: int = 120) -> dict:
    from openai import OpenAI
    key = _key()
    if not key: return {"ok": False, "err": "no zen key"}
    model = model or DEFAULTS[0]
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    try:
        c = OpenAI(api_key=key, base_url=BASE,
                   default_headers={"x-opencode-session": str(uuid.uuid4()),
                                    "User-Agent": "opencode/0.20.5",
                                    "HTTP-Referer": "https://opencode.ai/", "X-Title": "opencode"})
        r = c.chat.completions.create(model=model, messages=msgs, timeout=timeout)
        return {"ok": True, "model": model, "answer": r.choices[0].message.content or ""}
    except Exception as e:
        return {"ok": False, "model": model, "err": str(e)[:300]}
