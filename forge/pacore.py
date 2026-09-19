"""PaCoRe-inspired parallel coordinated reasoning for FORGE-4B.

Paper: PaCoRe (ACL 2026 / arXiv 2601.05593) — Parallel Coordinated Reasoning.
Idea: R rounds, K parallel trajectories per round, compact to messages, synthesize.

Lightweight adaptation for 4B laptop (no RL, inference-only):
- Round 1: K=4 parallel forks (temp/persona diverse, sequentially to avoid OOM)
- Compact: verifier verdict + 150-char per candidate -> message (~600 tokens)
- Round 2: synthesize message + problem -> K diverse refinements
- Final: verifier-weighted vote; if any ok=True, pick shortest ok.

This decouples TTC from context: 8*2048 TTC ~ 16k effective without 16k ctx.
"""
from __future__ import annotations
import re
from .student import Student
from .verifier import verify_python, check_constraints
from .swarm import BIGPICKLE_CRITIC_SYS

PERSONAS = ["coder", "reasoner", "critic", "researcher"]
TEMPS = [0.7, 0.9, 0.5, 0.85]

def _check(item: dict, answer: str) -> dict:
    m = re.findall(r"```python(.*?)```", answer or "", re.S)
    code = m[0].strip() if m else ""
    if item.get("tests") and code:
        v = verify_python(code, item["tests"])
        if v.get("ok") and item.get("answer_re"):
            if not re.search(item["answer_re"], answer, re.S):
                return {"ok": False, "log": "tests-pass-but-tag-missing"}
        return v
    if item.get("tests") and not code:
        return {"ok": False, "stage": "empty", "log": "no code block"}
    if item.get("must_include") is not None or item.get("answer_re"):
        if item.get("answer_re"):
            return {"ok": bool(re.search(item["answer_re"], answer, re.S)), "log": "regex"}
        return check_constraints(answer, item.get("must_include", []), item.get("exclude", []), item.get("format", ""))
    return {"ok": True, "log": "no-check"}

def pacore_solve(problem: str, item: dict, model: str = "kamekichi128/qwen3-4b-instruct-2507:latest", K: int = 4, R: int = 2) -> tuple[str, list]:
    """Returns (final_answer, trace). Trace keeps all attempts for bench."""
    s = Student(model)
    trace = []
    # Round 1: parallel forks
    round1 = []
    for i in range(K):
        persona_sys = f"You are a {PERSONAS[i % 4]} specialist. Be concise."
        try:
            ans = s.generate(problem, system=persona_sys, temperature=TEMPS[i % 4])
        except Exception as e:
            ans = f"[fork {i} failed: {e}]"
        v = _check(item, ans)
        round1.append((ans, v))
        trace.append({"round": 1, "fork": i, "ok": bool(v.get("ok")), "log": str(v)[:120]})
        # early exit if any passes and short
        if v.get("ok") and len(ans) < 800:
            pass
    # Compact -> message
    compact = []
    for i, (ans, v) in enumerate(round1):
        snippet = ans[:300].replace("\n", " ")
        compact.append(f"[{i}] ok={v.get('ok')} log={str(v)[:80]} ans={snippet}")
    message = "\n".join(compact)
    # Check if any round1 already passes -> can return best
    passed = [ans for ans, v in round1 if v.get("ok")]
    if passed:
        # pick shortest passed (anti-verbosity, per PaCoRe synthesis)
        best = min(passed, key=len)
        # Still do synthesis round to try to beat it, but we have a fallback
        fallback = best
    else:
        fallback = None

    if R < 2:
        if fallback:
            return fallback, trace
        # vote: pick most common answer length bucket
        best = min([a for a, _ in round1], key=len)
        return best, trace

    # Round 2: synthesize message to guide refinement
    synth_prompt = (f"Problem: {problem}\n\nRound1 compact ({K} parallel attempts):\n{message}\n\n"
                    "Synthesize: pick the correct approach, fix errors seen in logs, and produce final answer. "
                    "For code tasks, output ```python block. Be concise. Verify before finalizing.")
    round2 = []
    for i in range(K):
        try:
            ans = s.generate(synth_prompt, temperature=0.6 + 0.1*(i%2))
        except Exception as e:
            ans = f"[r2 fork {i} failed: {e}]"
        v = _check(item, ans)
        round2.append((ans, v))
        trace.append({"round": 2, "fork": i, "ok": bool(v.get("ok")), "log": str(v)[:120]})
    # Final vote: verifier-weighted shortest ok, else shortest overall
    all_cands = round1 + round2
    ok_cands = [(a, v) for a, v in all_cands if v.get("ok")]
    if ok_cands:
        final = min([a for a, _ in ok_cands], key=len)
        return final, trace
    # No ok: try one more synthesis with explicit verifier hint
    hint = "Previous attempts failed verifier. Check constraints, include code block, ensure tests pass."
    try:
        ans = s.generate(problem + "\n" + hint + "\nCompact: " + message[:600], temperature=0.5)
        trace.append({"round": 3, "fork": 0, "ok": bool(_check(item, ans).get("ok")), "log": "final-hint"})
        return ans, trace
    except Exception:
        return round1[0][0], trace
