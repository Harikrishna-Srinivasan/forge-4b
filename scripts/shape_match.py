"""Per-donor shape verdict WITHOUT downloading weights (config.json only, KBs).

Usage: python3 scripts/shape_match.py deepseek-ai/DeepSeek-R1 openai/gpt-oss-120b
Verdict classes: GRAFTABLE (all dims match) / DISTILL-ONLY (outputs transfer, weights don't).
"""
from __future__ import annotations
import json, sys
from huggingface_hub import hf_hub_download

BASE_CFG = {"architectures": ["Qwen3ForCausalLM"], "hidden_size": 2560,
            "num_hidden_layers": 36, "num_attention_heads": 32,
            "num_key_value_heads": 8, "intermediate_size": 9728, "vocab_size": 151936}

DIMS = ("hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size")


def verdict(repo: str) -> dict:
    try:
        c = json.loads(open(hf_hub_download(repo, "config.json")).read())
    except Exception as e:
        return {"repo": repo, "verdict": "UNKNOWN (gated/unreachable)", "detail": str(e)[:120]}
    diffs = [k for k in DIMS if c.get(k) != BASE_CFG[k]]
    arch_same = (c.get("architectures") or ["?"])[0] == BASE_CFG["architectures"][0]
    moe = "moe" in str(c.get("architectures")).lower() or c.get("num_experts") is not None
    if arch_same and not diffs and not moe:
        v = "GRAFTABLE (same-arch dense: linear/TIES/DARE via forge.merge.stream_merge)"
    else:
        why = []
        if not arch_same:
            why.append(f"arch {c.get('architectures')}")
        if diffs:
            why.append("dims differ: " + ",".join(diffs))
        if moe:
            why.append("MoE-vs-dense (no experts in base)")
        v = "DISTILL-ONLY (" + "; ".join(why) + ")"
    return {"repo": repo, "verdict": v,
            "arch": (c.get("architectures") or ["?"])[0]}


if __name__ == "__main__":
    for r in sys.argv[1:]:
        d = verdict(r)
        print(f"{d['repo']}\n  arch={d.get('arch')}\n  -> {d['verdict']}")
