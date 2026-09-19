"""Ollama student wrapper. Default: kamekichi128/qwen3-4b-instruct-2507 (Qwen3-4B-Instruct-2507,
4.0B, Q4_K_M, 262k ctx, Apache-2.0). hari-ai (=openbmb/minicpm5-2b, ~2B) usable as 2nd teacher.
No fine-tuning. Inference-only."""
from __future__ import annotations
import json, os, urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("FORGE_STUDENT", "kamekichi128/qwen3-4b-instruct-2507:latest")

SYSTEM_COMPACT = (
 "You are FORGE-4B student (<=4B). Use compact structured reasoning:\n"
 "OBSERVATION / HYPOTHESIS / EVIDENCE / ACTION / RESULT / UPDATED_BELIEF / VERIFICATION.\n"
 "Be concise. Never hallucinate APIs. Verify with tools. If uncertain, say what test would decide it."
)

def chat(model: str, messages: list, temperature: float = 0.7, top_p: float = 0.8,
         top_k: int = 40, num_ctx: int = 4096, num_predict: int = 300, think: bool = False) -> str:
    # qwen3:4b is thinking-capable; disable thinking for agentic speed unless needed
    msgs = list(messages)
    if not think and model.startswith("qwen3"):
        # ollama convention: /no_think suffix in user prompt
        if msgs and msgs[-1]["role"] == "user" and "/no_think" not in msgs[-1]["content"]:
            msgs[-1] = {"role": msgs[-1]["role"], "content": msgs[-1]["content"] + "\n/no_think"}
    payload = {
        "model": model, "messages": msgs, "stream": False, "keep_alive": "5m",
        "think": think,
        "options": {"temperature": temperature, "top_p": top_p, "top_k": top_k, "num_ctx": num_ctx,
                    "num_predict": num_predict},
    }
    req = urllib.request.Request(
        OLLAMA_HOST + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["message"]["content"]

class Student:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
    def generate(self, prompt: str, system: str = SYSTEM_COMPACT,
                 temperature: float = 0.7, top_p: float = 0.8, top_k: int = 40,
                 think: bool = False, num_predict: int = 300) -> str:
        return chat(self.model,
                    [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    temperature=temperature, top_p=top_p, top_k=top_k, think=think, num_predict=num_predict)
