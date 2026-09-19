"""03_merge_run.py — same-arch graft on the GPU box (needs ~25GB disk, ~6GB RAM).
Run: python colab/03_merge_run.py --donor mini-coder-4b --method ties --out models/forge-4b-ties
Then SFT *that* dir: python colab/02_sft.py --out models/forge-4b-ties-code
(reuses forge.merge.stream_merge; laptop-safe streaming, GPU-box-fast)."""
from __future__ import annotations
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default="mini-coder-4b")
    ap.add_argument("--method", default="ties", choices=["linear", "ties", "dare"])
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--out", default="models/forge-4b-merged")
    a = ap.parse_args()
    from forge.merge.stream_merge import run
    from forge.donors import BASE, DONORS
    print(run(BASE, DONORS[a.donor]["repo"], a.method, a.t, 0.2, 0.1, 0, a.out,
              "^(model.embed_tokens|lm_head)", 2.0))


if __name__ == "__main__":
    main()
