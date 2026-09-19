"""01_harvest.py — pull teacher OUTPUTS (never weights) into datasets/.
Run: python colab/01_harvest.py [--n 2000] [--include-b]
Default variant A only (permissive). --include-b adds quarantined Opus distills."""
from __future__ import annotations
import argparse, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

A_SOURCES = [
    ("Jackrong/gpt-oss-120B-distilled-reasoning", 50),
    ("Jackrong/Qwen3.5-reasoning-700x", 50),
]
B_SOURCES = [
    ("nohurry/Opus-4.6-Reasoning-3000x-filtered", 50),
]


def harvest_one(ds_name: str, n: int) -> None:
    import json
    from datasets import load_dataset
    outdir = os.path.join(ROOT, "datasets")
    os.makedirs(outdir, exist_ok=True)
    B_MARKERS = ("opus", "claude", "anthropic")
    ds = load_dataset(ds_name, split="train")
    lic = str(getattr(ds, "info", None) and ds.info.license or "unknown-do-not-redistribute")
    variant = "B-quarantined" if any(m in (ds_name + " " + lic).lower() for m in B_MARKERS) else "A-permissive"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ds_name)
    out = open(os.path.join(outdir, f"harvest-{safe}.jsonl"), "w")
    kept = 0
    for row in ds:
        if kept >= n:
            break
        keys = {k.lower(): k for k in row.keys()}
        prob = row.get(keys.get("input", keys.get("problem", ""))) if keys else ""
        ans = row.get(keys.get("output", keys.get("answer", ""))) if keys else ""
        if not prob or not ans:
            strs = [v for v in row.values() if isinstance(v, str) and len(v) > 10]
            if len(strs) < 2:
                continue
            prob, ans = strs[0], strs[1]
        out.write(json.dumps({"problem": str(prob)[:4000], "answer": str(ans)[:8000],
                              "teacher": ds_name, "license": lic, "variant": variant}) + "\n")
        kept += 1
    print(f"harvested {kept} -> {out.name} (variant={variant}, license={lic})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="cap per source (0=use tiny 50 for smoke test)")
    ap.add_argument("--include-b", action="store_true")
    a = ap.parse_args()
    srcs = list(A_SOURCES) + (B_SOURCES if a.include_b else [])
    for repo, n in srcs:
        harvest_one(repo, a.n or n)
    print("then: python scripts/build_manifest.py datasets datasets/manifest.jsonl")


if __name__ == "__main__":
    main()
