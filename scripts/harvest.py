"""Output-harvester: take big models' essential part (reasoning trajectories) WITHOUT
downloading weights. HF dataset (MBs) -> forge SFT jsonl with provenance/quarantine tags.

Usage: python3 scripts/harvest.py Jackrong/gpt-oss-120B-distilled-reasoning 50
Writes datasets/harvest-<name>.jsonl, one record per sample:
  {problem, answer, teacher, license, variant}
"""
from __future__ import annotations
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "datasets")
B_MARKERS = ("opus", "claude", "anthropic")


def variant(name: str, lic: str) -> str:
    s = (name + " " + lic).lower()
    return "B-quarantined" if any(m in s for m in B_MARKERS) else "A-permissive"


def main():
    ds_name, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
    from datasets import load_dataset
    ds = load_dataset(ds_name, split="train")
    lic = str(getattr(ds, "info", None) and ds.info.license or "unknown-do-not-redistribute")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ds_name)
    os.makedirs(OUTDIR, exist_ok=True)
    out = open(os.path.join(OUTDIR, f"harvest-{safe}.jsonl"), "w")
    kept = 0
    for row in ds:
        if kept >= n:
            break
        keys = {k.lower(): k for k in row.keys()}
        prob = row.get(keys.get("input", keys.get("problem", ""))) if keys else ""
        ans = (row.get(keys.get("output", keys.get("answer", ""))) if keys else "")
        if not prob or not ans:
            # fallback: first two string fields
            strs = [v for v in row.values() if isinstance(v, str) and len(v) > 10]
            if len(strs) < 2:
                continue
            prob, ans = strs[0], strs[1]
        out.write(json.dumps({"problem": str(prob)[:4000], "answer": str(ans)[:8000],
                              "teacher": ds_name, "license": lic,
                              "variant": variant(ds_name, lic)}) + "\n")
        kept += 1
    print(f"harvested {kept} -> {out.name} (variant={variant(ds_name, lic)}, license={lic})")


if __name__ == "__main__":
    main()
