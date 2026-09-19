"""Orchestrator: task -> generator -> 3-5 teachers -> judge -> executor -> judge again -> dataset builder.
Inference-only: teachers are diverse local samples from qwen3:4b (no fine-tune)."""
from __future__ import annotations
import re, time
from .teachers import TeacherRouter
from .verifier import verify_python, check_constraints, contradiction_scan, judge_pick
from .student import Student

CODE_RE = re.compile(r"```python(.*?)```", re.S)

def extract_code(answer: str) -> str:
    m = CODE_RE.findall(answer or "")
    if m: return m[0].strip()
    # fallback: lines that look like python defs
    lines = [l for l in (answer or "").splitlines()
             if l.strip().startswith(("def ", "class ", "import ", "from ", "return "))]
    return "\n".join(lines)

class Orchestrator:
    def __init__(self, model: str = "kamekichi128/qwen3-4b-instruct-2507:latest"):
        self.router = TeacherRouter(model)
        self.judge_llm = Student(model)
        self.model = model

    def run(self, problem: str, capability: str = "04_coding",
            tests: str = "", constraints: dict | None = None, max_repair: int = 2) -> dict:
        t0 = time.time()
        trajectory = []
        # 1. generate teachers
        cands = self.router.ask_teachers(problem, capability, n=3)
        # 2. first judge: deterministic verify each candidate
        verifs = []
        for c in cands:
            v: dict = {"ok": None}
            code = extract_code(c["answer"])
            if code and capability in ("04_coding", "05_debugging"):
                v = verify_python(code, tests)
            if constraints:
                v2 = check_constraints(c["answer"], constraints.get("include", []),
                                       constraints.get("exclude", []), constraints.get("format", ""))
                v = {**v, **v2, "ok": bool(v.get("ok", True)) and v2["ok"]} if v.get("ok") is not None else v2
            contra = contradiction_scan(c["answer"])
            v["contradiction_flags"] = contra["flags"]
            if v.get("ok") is None:
                v["ok"] = not contra["flags"]
            verifs.append(v)
        best = judge_pick(cands, verifs)
        best_ans, best_ver = cands[best]["answer"], verifs[best]
        trajectory.append({"stage": "first-judge", "best": best, "verifs": verifs})
        # 3. executor + repair loop (TIME_TO_ERROR_DETECTION: repair ASAP)
        code = extract_code(best_ans)
        repair_log = []
        for attempt in range(max_repair + 1):
            if code and capability in ("04_coding", "05_debugging"):
                v = verify_python(code, tests)
                repair_log.append({"attempt": attempt, "ok": v["ok"], "log": v["log"][:500]})
                if v["ok"]:
                    best_ver = v; break
                # repair with critic prompt
                fix_prompt = (f"Problem: {problem}\nFailing code:\n{code}\nTest log:\n{v['log']}\n"
                              "OBSERVATION/HYPOTHESIS/EVIDENCE/ACTION/RESULT/UPDATED_BELIEF/VERIFICATION. "
                              "Return fixed ```python block only, minimal change./no_think")
                try:
                    repaired = self.judge_llm.generate(fix_prompt, temperature=0.4)
                    code = extract_code(repaired) or code
                    best_ans = repaired
                except Exception as e:
                    repair_log.append({"repair_failed": str(e)}); break
            else:
                break
        # 4. final judge again
        out = {"problem": problem, "capability": capability, "answer": best_ans,
               "verification": best_ver, "candidates": cands, "repair_log": repair_log,
               "trajectory": trajectory, "model": self.model,
               "elapsed_s": round(time.time() - t0, 2), "no_finetune": True}
        return out
