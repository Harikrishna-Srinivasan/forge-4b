"""7-teacher routing -> 4B student. No fine-tune locally; teachers give trajectories/critiques
used as supervision (exported for remote SFT/DPO/RL). Never assume license; provenance always kept."""
from __future__ import annotations
import os
from .student import Student, SYSTEM_COMPACT, chat

PERSONAS = {
 "coder": "You are a senior coding agent. Output code first, then 3-line diagnosis. Verify by mental execution.",
 "reasoner": "You are a math/reasoning specialist. Derive step by step, check each step, flag unsupported claims.",
 "critic": "You are an adversarial critic. Find the earliest wrong assumption, give a counterexample if any.",
 "researcher": "You are a web-research planner. List what evidence is needed and how to obtain it.",
 "debugger": "You are a debugger. Propose symptom->hypotheses->discriminating test->patch->regression test.",
}

TEACHERS_7 = {
 "deepseek": {"strengths": ["02_reasoning", "03_math", "04_coding", "05_debugging"],
              "local_ollama": "deepseek-r1:1.5b", "api_env": "DEEPSEEK_API_KEY",
              "size_hint": "1.5B-local or R1/V3-API", "license_note": "verify card per release; do-not-redistribute until checked"},
 "glm": {"strengths": ["02_reasoning", "04_coding", "09_tool_use", "01_instruction"],
         "local_ollama": None, "api_env": "GLM_API_KEY",
         "size_hint": "GLM-4/4.5-API (local 9B too big for 7.5GB laptop)",
         "license_note": "Zhipu custom; verify; research-use-only until checked"},
 "qwen": {"strengths": ["04_coding", "05_debugging", "06_code_localization", "11_long_context", "09_tool_use"],
          "local_ollama": "kamekichi128/qwen3-4b-instruct-2507:latest", "api_env": "QWEN_API_KEY",
          "size_hint": "4.0B-local Apache-2.0 verified (exact Instruct-2507); larger 30B/235B via API",
          "license_note": "Apache-2.0 (Qwen3-4B-Instruct-2507 HF card verified)"},
 "mimo": {"strengths": ["02_reasoning", "03_math", "04_coding"],
          "local_ollama": None, "api_env": "MIMO_API_KEY",
          "size_hint": "MiMo-7B-API (verify)", "license_note": "verify Xiaomi card before redistribute"},
 "kimi": {"strengths": ["11_long_context", "10_web_research", "07_repository_reasoning"],
          "local_ollama": None, "api_env": "KIMI_API_KEY",
          "size_hint": "Kimi-K2-API long-context", "license_note": "verify Moonshot card before redistribute"},
 "minimax": {"strengths": ["16_general_instruction", "01_instruction", "10_web_research"],
             "local_ollama": None, "api_env": "MINIMAX_API_KEY",
             "size_hint": "MiniMax-API", "license_note": "verify MiniMax card before redistribute"},
 "gpt-oss": {"strengths": ["05_debugging", "04_coding", "13_contradiction_detection", "14_agent_recovery"],
             "local_ollama": None, "api_env": "GPTOSS_API_KEY",
             "size_hint": "gpt-oss-20b/120b-API (20B too big for 7.5GB laptop)",
             "license_note": "Apache-2.0 per gpt-oss card; verify release"},
}

PERSONA_FOR_TEACHER = {"deepseek": "reasoner", "glm": "coder", "qwen": "coder",
                       "mimo": "reasoner", "kimi": "researcher", "minimax": "researcher",
                       "gpt-oss": "critic"}

def pick_teachers(capability: str, n: int = 4) -> list[str]:
    scored = [(1 if capability in info["strengths"] else 0, name)
              for name, info in TEACHERS_7.items()]
    scored.sort(reverse=True)
    picks = [nm for s, nm in scored if s == 1][:n]
    for _, nm in scored:
        if len(picks) >= max(3, min(5, n)): break
        if nm not in picks: picks.append(nm)
    return picks[:max(3, min(5, n))]

def _ollama_installed(model: str | None) -> bool:
    if not model: return False
    try:
        import urllib.request, json
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as r:
            tags = json.loads(r.read().decode()).get("models", [])
        names = [m.get("name", "") for m in tags]
        return any(model in n for n in names)
    except Exception:
        return False

class TeacherRouter:
    """Route to 7 teachers. Local ollama if installed, else persona-diverse Instruct-2507 samples.
    This is how DeepSeek+GLM+Qwen+MiMo+Kimi+MiniMax+GPT-OSS knowledge is compressed
    into supervision for the 4B student WITHOUT local fine-tuning."""
    def __init__(self, student_model: str = "kamekichi128/qwen3-4b-instruct-2507:latest"):
        self.local = Student(student_model)
        self.model_name = student_model

    def route(self, capability: str) -> list[str]:
        table = {
         "04_coding": ["coder", "debugger", "critic"],
         "05_debugging": ["debugger", "coder", "critic"],
         "02_reasoning": ["reasoner", "critic", "coder"],
         "03_math": ["reasoner", "critic"],
         "13_contradiction_detection": ["critic", "reasoner"],
         "06_code_localization": ["coder", "critic"],
         "09_tool_use": ["coder", "researcher", "critic"],
        }
        return table.get(capability, ["coder", "reasoner", "critic"])

    def ask_teachers(self, problem: str, capability: str, n: int = 4,
                     temperature: float = 0.7) -> list[dict]:
        teacher_names = pick_teachers(capability, n)
        personas = self.route(capability)
        out = []
        for i, tname in enumerate(teacher_names):
            info = TEACHERS_7[tname]
            persona = PERSONA_FOR_TEACHER[tname]
            sys = SYSTEM_COMPACT + "\n" + PERSONAS[persona] + f"\nEmulate {tname} strength: {','.join(info['strengths'][:3])}."
            local_model = info["local_ollama"]
            use_model = local_model if _ollama_installed(local_model) else self.model_name
            t = [0.7, 0.9, 0.5, 0.8, 0.6][i % 5]
            try:
                if use_model == self.model_name:
                    ans = self.local.generate(problem, system=sys, temperature=t)
                else:
                    ans = chat(use_model, [{"role": "system", "content": sys},
                                           {"role": "user", "content": problem + "\n/no_think"}],
                               temperature=t)
                src = f"{tname}-via-{use_model}"
            except Exception as e:
                ans, src = f"[{tname} failed: {e}]", f"{tname}-failed"
            out.append({"agent": src, "teacher_family": tname, "answer": ans,
                        "persona": persona, "license": info["license_note"],
                        "teacher_size": info["size_hint"]})
        # file-based teacher trajectories (SWE-smith / mini-coder / Jackrong etc.) if dropped in data/
        out.extend(self._file_teachers()[:1])
        return out

    def _file_teachers(self) -> list[dict]:
        import glob
        hits = []
        for p in glob.glob("/home/dell/forge-4b/data/*.jsonl")[:2]:
            try:
                line = open(p).readline().strip()
                if line:
                    import json
                    d = json.loads(line)
                    hits.append({"agent": f"file:{p}", "teacher_family": d.get("teacher", "file"),
                                 "answer": d.get("answer", "")[:2000], "persona": "coder",
                                 "license": d.get("license", "unknown-do-not-redistribute"),
                                 "teacher_size": d.get("teacher_size", "unknown")})
            except Exception:
                pass
        return hits
