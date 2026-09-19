"""Automated loop: TASK POOL -> TEACHER SWARM -> DISAGREEMENT -> CRITIC -> EXECUTOR
-> VERIFIED RESULT -> SUCCESS/dataset | FAILURE/lesson -> EVALUATE -> WEAKNESS -> TARGETED TASKS -> LOOP.
On-policy: student attempts first; teacher tutors from stuck state. Laptop-only, no training."""
from __future__ import annotations
import json, os, re, time
from .swarm import Swarm, make_failure_lesson
from .verifier import verify_python, check_constraints

BASE = "/home/dell/forge-4b"

def load_tasks(limit: int = 2) -> list[dict]:
    out = []
    p = os.path.join(BASE, "tasks", "seed.jsonl")
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
        if len(out) >= limit: break
    return out

def extract_code(ans: str) -> str:
    m = re.findall(r"```python(.*?)```", ans or "", re.S)
    if m: return m[0].strip()
    return ""

def safe_call(fn, *a, **k) -> dict:
    try: return {"ok_call": True, "res": fn(*a, **k)}
    except Exception as e: return {"ok_call": False, "err": str(e)[:300]}

def run_pipeline(limit: int = 2) -> dict:
    sw = Swarm()  # single-model default (qwen3:4b) avoids OOM on 7.5GB laptop
    summary = {"tasks": [], "weakness": {}, "n_verified": 0, "n_lessons": 0}
    for t in load_tasks(limit):
        r1 = safe_call(sw.student_attempt, t["prompt"] + " /no_think")
        s = r1["res"] if r1["ok_call"] else {"agent": "student-failed", "answer": ""}
        if r1["ok_call"]:
            r2 = safe_call(sw.mimo_solve, f"Tutor full solution. Task: {t['prompt']} /no_think")
        else:
            r2 = {"ok_call": False}
        m = r2["res"] if isinstance(r2, dict) and r2.get("ok_call") else {"agent": "teacher-failed", "answer": ""}
        cands = [s, m]
        r3 = safe_call(sw.bigpickle_critique, t["prompt"], cands)
        crit = r3["res"] if r3.get("ok_call") else {"verdict": "critic-skipped-OOM", "disagreement": None}
        # executor: verify EACH candidate, pick best (judge again after execution)
        scored = []
        for c in cands:
            code = extract_code(c.get("answer", ""))
            if code and t.get("tests"): v = verify_python(code, t["tests"])
            elif t["type"] == "instruction":
                v = check_constraints(c.get("answer", ""), ["kernel"] if "kernel" in t["prompt"] else [], ["windows"])
            else: v = {"ok": None, "note": "no deterministic check"}
            scored.append((c, code, v))
        # prefer ok==True, then non-empty code, then shorter answer (anti-verbosity)
        scored.sort(key=lambda x: (1 if x[2].get("ok") else 0, 1 if x[1] else 0, -len(x[0].get("answer", ""))))
        best_c, best_code, ver = scored[-1]
        ok = bool(ver.get("ok"))
        rec = {"id": t["id"], "type": t["type"], "ok": ok, "verification": ver,
               "disagreement": crit.get("disagreement"), "best_from": best_c.get("agent"),
               "critic": str(crit.get("verdict", ""))[:600]}
        ts = time.strftime("%Y%m%d-%H%M%S")
        if ok:
            open(f"{BASE}/verified/{t['id']}-{ts}.json", "w").write(json.dumps(
                {"task": t, "student": s, "teacher": m, "best": best_c, "critic": crit,
                 "verification": ver}, indent=2)[:8000])
            # SAVE BEST VERIFIED ANSWER (fix: was saving teacher prose before)
            best_ans = best_code if best_code else best_c.get("answer", "")[:1500]
            open(f"{BASE}/datasets/forge-sft.jsonl", "a").write(json.dumps(
                {"problem": t["prompt"], "answer": best_ans,
                 "teacher": best_c.get("agent", ""),
                 "license": best_c.get("license", "Apache-2.0(local)")
                 if "zen-cloud" not in str(best_c.get("agent", ""))
                 else "unknown-do-not-redistribute(zen-cloud)",
                 "capability": t["type"]}) + "\n")
            summary["n_verified"] += 1
        else:
            lesson = make_failure_lesson(t, bad_approach=s.get("answer", "")[:400],
                                         observation=str(ver)[:400],
                                         correction=m.get("answer", "")[:400],
                                         lesson="Detect error early; verify with execution, not confidence.")
            open(f"{BASE}/failures/{t['id']}-{ts}.json", "w").write(json.dumps(lesson, indent=2))
            open(f"{BASE}/datasets/forge-lessons.jsonl", "a").write(json.dumps(lesson) + "\n")
            summary["n_lessons"] += 1
        open(f"{BASE}/trajectories/{t['id']}-{ts}.json", "w").write(json.dumps(rec, indent=2))
        summary["tasks"].append(rec)
        time.sleep(2)  # let ollama unload between tasks on small RAM
    fails = [r for r in summary["tasks"] if not r["ok"]]
    summary["weakness"] = {"failed_types": [r["type"] for r in fails],
                           "next": f"generate 1000x {fails[0]['type']}" if fails else "scale to 1000 tasks x3 teachers"}
    open(f"{BASE}/evaluations/eval-{time.strftime('%Y%m%d-%H%M%S')}.json", "w").write(
        json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    import sys
    print(json.dumps(run_pipeline(int(sys.argv[1]) if len(sys.argv) > 1 else 2), indent=2))
