"""FORGE Arena: 50Q max, 5 attempts each till PASS. No training/weight-share/RL.
Two tracks: A=tools+verifier (our strength), B=no-tool single-shot (ablation).
Survivorship-safe: report pass@1 AND pass@5, all attempts logged, never filter to wins.
Defensive security only, isolated sandbox."""
from __future__ import annotations
import json, os, re, time
from .swarm import Swarm
from .verifier import verify_python, check_constraints, contradiction_scan

BASE = "/home/dell/forge-4b"
ARENA = os.path.join(BASE, "tasks", "arena50.jsonl")

def extract_code(a: str) -> str:
    m = re.findall(r"```python(.*?)```", a or "", re.S)
    return m[0].strip() if m else ""

def check(item: dict, answer: str) -> dict:
    code = extract_code(answer)
    if item.get("tests") and code:
        v = verify_python(code, item["tests"])
        # enforce BOTH code-tests and answer tag when item has both (e.g. LON-30)
        if v.get("ok") and item.get("answer_re"):
            if not re.search(item["answer_re"], answer, re.S):
                return {"ok": False, "log": "tests-pass-but-tag-missing"}
        return v
    # PaCoRe track C uses same checks (extracted for reuse)
    if item.get("answer_re") and not item.get("tests"):
        return {"ok": bool(re.search(item["answer_re"], answer, re.S)), "log": "regex"}
    if item.get("must_include") is not None:
        return check_constraints(answer, item.get("must_include", []), item.get("exclude", []), item.get("format", ""))
    if item.get("tests") and not code:
        return {"ok": False, "stage": "empty", "log": "no code block"}
    if item.get("must_include") is not None:
        return check_constraints(answer, item.get("must_include", []),
                                 item.get("exclude", []), item.get("format", ""))
    if item.get("answer_re"):
        return {"ok": bool(re.search(item["answer_re"], answer, re.S)),
                "log": "regex-check"}
    c = contradiction_scan(answer)
    return {"ok": not c["flags"], "flags": c["flags"]}

def run_arena(limit: int = 50, attempts: int = 5, track: str = "A",
              model: str = "kamekichi128/qwen3-4b-instruct-2507:latest", pacore_K: int = 4, pacore_R: int = 2) -> dict:
    # Track C = PaCoRe coordinated (fork-join, R rounds, K parallel)
    if track == "C":
        from .pacore import pacore_solve
        items = [json.loads(l) for l in open(ARENA) if l.strip()][:limit]
        out = {"track": track, "model": model, "results": [], "pass1": 0, "pass5": 0, "n": 0, "pacore": f"K={pacore_K} R={pacore_R}"}
        ap = os.path.join(BASE, "evaluations", "arena.jsonl")
        for it in items:
            try:
                final, trace = pacore_solve(it["prompt"], it, model, pacore_K, pacore_R)
                v = check(it, final)
                ok = bool(v.get("ok"))
            except Exception as e:
                final, trace, ok, v = "", [], False, {"log": str(e)[:200]}
            out["n"] += 1
            out["pass1"] += ok
            out["pass5"] += ok
            rec = {"id": it["id"], "aspect": it["aspect"], "final_ok": ok, "pass1": ok, "attempts": trace[:8], "verdict": str(v)[:200]}
            open(ap, "a").write(json.dumps(rec) + "\n")
            out["results"].append(rec)
            time.sleep(1)
        by = {}
        for r in out["results"]:
            by.setdefault(r["aspect"], [0, 0])
            by[r["aspect"]][1] += 1; by[r["aspect"]][0] += r["final_ok"]
        out["by_aspect"] = {k: f"{a}/{b}" for k, (a, b) in by.items()}
        out["note"] = "PaCoRe track C: K parallel forks per round, compact+synthesize. Effective TTC >> context."
        open(os.path.join(BASE, "evaluations", f"arena-{track}-{time.strftime('%H%M%S')}.json"), "w").write(json.dumps(out, indent=2))
        return out
    sw = Swarm(model, model)
    items = [json.loads(l) for l in open(ARENA) if l.strip()][:limit]
    # resume: skip ids already in evaluations/arena.jsonl with ok True
    done = set()
    ap = os.path.join(BASE, "evaluations", "arena.jsonl")
    if os.path.exists(ap):
        for l in open(ap):
            try:
                d = json.loads(l)
                if d.get("final_ok"): done.add(d["id"])
            except Exception: pass
    out = {"track": track, "model": model, "results": [], "pass1": 0, "pass5": 0, "n": 0}
    for it in items:
        if it["id"] in done: continue
        tries, ok1, ok5, att_log = [], False, False, []
        for a in range(attempts):
            use_tool = (track == "A")  # track B: single-shot, no repair/executor hint
            try:
                r = sw.student_attempt(it["prompt"] + " /no_think")
                ans = r["answer"]
            except Exception as e:
                att_log.append({"attempt": a, "ok": False, "err": str(e)[:200]}); continue
            v = check(it, ans)
            ok = bool(v.get("ok"))
            att_log.append({"attempt": a, "ok": ok, "log": str(v)[:300],
                            "answer": ans[:600]})  # survivorship fix: keep losers' text
            tries.append(ans[:500])
            if a == 0 and ok: ok1 = True
            if ok:
                ok5 = True
                if use_tool and a < attempts - 1 and not ok: pass
                break
            # ReAct repair: feed tool log back (track A only)
            if use_tool:
                try:
                    fix = sw.mimo_solve(f"Task: {it['prompt']}\nFailed answer: {ans[:800]}\nCheck: {v}\nFix minimally. ```python block if code./no_think")
                    ans2 = fix["answer"]; v2 = check(it, ans2)
                    att_log.append({"attempt": f"{a}r", "ok": bool(v2.get("ok")), "log": str(v2)[:300],
                                    "answer": ans2[:600]})
                    if v2.get("ok"): ok5 = True; tries.append(ans2[:500]); break
                except Exception as e:
                    att_log.append({"repair_err": str(e)[:200]})
            time.sleep(1)
        out["n"] += 1
        out["pass1"] += ok1; out["pass5"] += ok5 or ok1
        rec = {"id": it["id"], "aspect": it["aspect"], "final_ok": ok5 or ok1,
               "pass1": ok1, "attempts": att_log}
        open(ap, "a").write(json.dumps(rec) + "\n")
        out["results"].append(rec)
        time.sleep(2)  # RAM breather
    # ASPECT breakdown + honesty footer
    by = {}
    for r in out["results"]:
        by.setdefault(r["aspect"], [0, 0])
        by[r["aspect"]][1] += 1; by[r["aspect"]][0] += r["final_ok"]
    out["by_aspect"] = {k: f"{a}/{b}" for k, (a, b) in by.items()}
    out["note"] = "pass@1=single-shot; pass@5=≤5 tries+repair. Report both. Never drop failures."
    # failure mining: export losers with answers (anti-survivorship: train on corrections, not wins)
    for r in out["results"]:
        if not r["final_ok"]:
            open(os.path.join(BASE, "datasets", "forge-lessons.jsonl"), "a").write(json.dumps({
                "id": r["id"], "aspect": r["aspect"], "attempts": r["attempts"],
                "lesson": "recovered" if any(a.get("ok") for a in r["attempts"]) else "all-failed-needs-targeted-gen",
                "license": "Apache-2.0(local)"}) + "\n")
    open(os.path.join(BASE, "evaluations", f"arena-{track}-{time.strftime('%H%M%S')}.json"), "w").write(json.dumps(out, indent=2))
    return out

if __name__ == "__main__":
    import sys
    print(json.dumps(run_arena(int(sys.argv[1]) if len(sys.argv) > 1 else 10,
                               int(sys.argv[2]) if len(sys.argv) > 2 else 5,
                               sys.argv[3] if len(sys.argv) > 3 else "A"), indent=2))
