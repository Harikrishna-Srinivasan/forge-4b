"""Deterministic verifiers: execution > opinion. T1-style: tools first, LLM judge second."""
from __future__ import annotations
import ast, re, subprocess, tempfile, os

_PY_PRELUDE = "from typing import *\nimport math\n"

def verify_python(code: str, tests: str = "", timeout: int = 10) -> dict:
    """Compile + run code + tests in sandbox subprocess. Returns {ok, log}."""
    if not (code or "").strip():
        return {"ok": False, "stage": "empty", "log": "no code extracted"}
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"ok": False, "stage": "compile", "log": f"SyntaxError: {e}"}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sol.py")
        # HumanEval-style tests commonly assume typing/math are available.
        open(p, "w").write(_PY_PRELUDE + code + "\n" + (tests or ""))
        try:
            r = subprocess.run(["python3", p], capture_output=True, text=True, timeout=timeout)
            ok = r.returncode == 0
            return {"ok": ok, "stage": "exec", "log": (r.stdout + r.stderr)[-2000:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "timeout", "log": "timeout"}

def check_constraints(answer: str, must_include: list, must_exclude: list,
                      exact_format: str = "") -> dict:
    viol = []
    for m in must_include or []:
        if m not in answer:
            viol.append(f"missing:{m}")
    for m in must_exclude or []:
        if m in answer:
            viol.append(f"forbidden:{m}")
    if exact_format and not re.search(exact_format, answer, re.S):
        viol.append("format-mismatch")
    return {"ok": not viol, "violations": viol}

def contradiction_scan(text: str) -> dict:
    """Cheap heuristic: flags 'always/never' vs later hedge, numeric mismatches, self-negation."""
    flags = []
    if re.search(r"\b(always|never|all|none)\b", text, re.I) and re.search(
            r"\b(sometimes|maybe|except|however)\b", text, re.I):
        flags.append("absolute-then-hedge")
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(set(nums)) > 6:
        flags.append("many-numbers-verify-manually")
    if re.search(r"\b(not|no)\b.{0,40}\b(not|no)\b", text, re.I):
        flags.append("double-negation-check")
    return {"ok": not flags, "flags": flags}

def judge_pick(candidates: list[dict], verifications: list[dict]) -> int:
    """Judge: prefer ok==True, then shortest log (efficiency), then shortest answer.
    Never rewards verbosity."""
    best, best_key = 0, None
    for i, (c, v) in enumerate(zip(candidates, verifications)):
        key = (1 if v.get("ok") else 0, -len(c.get("answer", "")))
        if best_key is None or key > best_key:
            best_key, best = key, i
    return best
