"""Src-alone offline bench. No network, no cloud, no training.
Enforces FORGE_ZEN=0 (local student only), runs Arena-50, compares vs stored
cloud baseline (46/50 pass@1, fails INS-04/REA-06/LON-30/SEC-45).
Run: FORGE_ZEN=0 python3 -m forge.bench.run [--limit 50] [--attempts 5] [--track A]
"""
from __future__ import annotations
import argparse, json, os, time

from ..offline import assert_offline

BASE = "/home/dell/forge-4b"
CLOUD_BASELINE = {"pass1": 46, "n": 50,
                  "fails": ["INS-04", "REA-06", "LON-30", "SEC-45"],
                  "note": "cloud-9B single-shot, same checkers, pasted 2026-09-13"}


def main():
    assert_offline()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--attempts", type=int, default=5)
    ap.add_argument("--track", default="A", choices=["A", "B", "C"])
    ap.add_argument("--model", default=os.environ.get(
        "FORGE_STUDENT", "kamekichi128/qwen3-4b-instruct-2507:latest"))
    a = ap.parse_args()
    os.environ["FORGE_ZEN"] = "0"
    from ..arena import run_arena
    res = run_arena(a.limit, a.attempts, a.track, a.model)
    ours_p1, ours_p5, n = res["pass1"], res["pass5"], res["n"]
    b = CLOUD_BASELINE
    verdict = ("PARITY/LEAD" if ours_p5 >= b["pass1"] and n >= 50
               else "BELOW (need pass@5>=46 on full 50)" if n >= 50
               else f"PARTIAL ({n}/50 run)")
    out = {"ts": time.strftime("%Y%m%d-%H%M%S"), "track": a.track, "model": a.model,
           "ours": {"pass1": ours_p1, "pass5": ours_p5, "n": n, "by_aspect": res["by_aspect"]},
           "cloud_baseline": b, "verdict": verdict}
    p = os.path.join(BASE, "evaluations", f"bench-{a.track}-{out['ts']}.json")
    open(p, "w").write(json.dumps(out, indent=2))
    print(f"ours pass@1={ours_p1}/{n} pass@5={ours_p5}/{n} | cloud pass@1={b['pass1']}/50")
    print(f"by_aspect: {json.dumps(res['by_aspect'])}")
    print(f"verdict: {verdict} -> {p}")


if __name__ == "__main__":
    main()
