"""Streaming same-arch merger: linear / TIES / DARE, per-tensor, laptop-RAM-safe.

Peak RAM ~1 shard (default 1GB) + 2 tensors. Full 4B BF16 dicts are NEVER held.
Usage:
  python3 -m forge.merge.stream_merge --dry-run --donor mini-coder-4b
  python3 -m forge.merge.stream_merge --method ties --t 0.5 --donor mini-coder-4b \\
      --out models/forge-4b-ties (run on GPU box; downloads ~16GB, writes ~8GB)
"""
from __future__ import annotations
import argparse, json, os, re
import torch

from ..donors import BASE, DONORS, fetch_config, compat_report


def _open(repo: str):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(repo, allow_patterns=["*.safetensors", "config.json"])
    files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    Join = lambda: None
    handles = [safe_open(os.path.join(d, f), framework="pt") for f in files]
    names = [k for h in handles for k in h.keys()]
    def get(n):
        for h in handles:
            if n in h.keys():
                return h.get_tensor(n)
        raise KeyError(n)
    return names, get


def merge_tensor(base: torch.Tensor, donor: torch.Tensor, method: str,
                 t: float, trim: float, drop: float, gen: torch.Generator) -> torch.Tensor:
    if base.dtype in (torch.float32, torch.float16, torch.bfloat16) and base.shape == donor.shape:
        b = base.float()
        tau = (donor.float() - b)
        if method == "linear":
            return ((1 - t) * b + t * donor.float()).to(base.dtype)
        if method == "ties":  # single-donor TIES: trim small-magnitude deltas, keep scale
            k = int(tau.numel() * trim)
            if k > 0:
                thr = tau.abs().flatten().kthvalue(k).values
                tau = torch.where(tau.abs() < thr, torch.zeros_like(tau), tau)
            return (b + tau).to(base.dtype)
        if method == "dare":  # dropout + rescale task vector
            mask = (torch.rand(tau.shape, generator=gen) > drop).float()
            return (b + mask * tau / (1 - drop)).to(base.dtype)
        raise ValueError(method)
    return base  # non-float / mismatched: keep base


def run(base_repo: str, donor_repo: str, method: str, t: float, trim: float,
        drop: float, seed: int, out: str, exclude: str, shard_gb: float):
    import safetensors.torch as st
    base_cfg, donor_cfg = fetch_config(base_repo), fetch_config(donor_repo)
    rep = compat_report(base_cfg, donor_cfg)
    if not rep["compatible"]:
        raise SystemExit(f"INCOMPATIBLE, refusing: {rep['diffs']}")
    names_b, get_b = _open(base_repo)
    _, get_d = _open(donor_repo)
    exc = re.compile(exclude) if exclude else None
    gen = torch.Generator().manual_seed(seed)
    os.makedirs(out, exist_ok=True)
    shard, shard_bytes, shards, idx, cap = {}, 0, [], {}, int(shard_gb * 1e9)
    def flush():
        nonlocal shard, shard_bytes
        fn = f"model-{len(shards)+1:05d}-of-00000.safetensors"
        st.save_file(shard, os.path.join(out, fn))
        for k, v in shard.items():
            idx[k] = fn
        shards.append(fn)
        shard, shard_bytes = {}, 0
    for n in names_b:
        b = get_b(n)
        if exc and exc.search(n):
            m = b
        else:
            try:
                m = merge_tensor(b, get_d(n), method, t, trim, drop, gen)
            except KeyError:
                m = b  # donor lacks tensor: keep base
        shard[n] = m
        shard_bytes += m.nbytes
        del b, m
        if shard_bytes >= cap:
            flush()
    if shard:
        flush()
    total = len(shards)
    for i, fn in enumerate(shards, 1):
        new = f"model-{i:05d}-of-{total:05d}.safetensors"
        os.rename(os.path.join(out, fn), os.path.join(out, new))
        for k in [k for k, v in idx.items() if v == fn]:
            idx[k] = new
    json.dump({"metadata": {}, "weight_map": idx},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    json.dump(base_cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(base_repo, f)
            import shutil
            shutil.copy(p, os.path.join(out, f))
        except Exception:
            pass
    return {"out": out, "shards": total, "method": method}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default="mini-coder-4b", choices=list(DONORS))
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--method", default="ties", choices=["linear", "ties", "dare"])
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--trim", type=float, default=0.2)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/forge-4b-merged")
    ap.add_argument("--exclude", default="^(model.embed_tokens|lm_head)")
    ap.add_argument("--shard-gb", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    donor_repo = DONORS[a.donor]["repo"]
    if a.dry_run:
        base_cfg, donor_cfg = fetch_config(a.base), fetch_config(donor_repo)
        print(json.dumps({"base": a.base, "donor": donor_repo, **compat_report(base_cfg, donor_cfg)}, indent=2))
        return
    print(json.dumps(run(a.base, donor_repo, a.method, a.t, a.trim, a.drop,
                         a.seed, a.out, a.exclude, a.shard_gb), indent=2))


if __name__ == "__main__":
    main()
