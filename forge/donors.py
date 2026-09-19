"""Same-architecture donor registry + compatibility checker.

RULE: grafting works ONLY between identical architectures (same hidden/layers/vocab/
attention). 'Active params' of an MoE are routing outputs, not transplantable
matrices — there is nothing to paste. Cross-family (GLM/DeepSeek/Kimi/MiMo/GPT-OSS
into Qwen3) is shape-incompatible, single-source or not.

Compatible donors for base Qwen/Qwen3-4B-Instruct-2507 (verified from model cards):
  - ricdomolm/mini-coder-4b ........ finetune OF Qwen3-4B-Instruct-2507 (coding/SWE)
  - CodeScout-4B ................... trained FROM Qwen3-4B-Instruct-2507 (code search)
Incompatible until proven otherwise (different arch family or size):
  - Jackrong Qwen3.5-* (Qwen3.5 arch != Qwen3 arch), MiniCPM, GLM, DeepSeek, Kimi, MoE A3B.
"""
from __future__ import annotations

BASE = "Qwen/Qwen3-4B-Instruct-2507"

DONORS = {
    "mini-coder-4b": {
        "repo": "ricdomolm/mini-coder-4b",
        "why": "SWE/coding finetune of our exact base; brings agentic-coding deltas",
        "license_note": "MIT card; verify file-level before redistribute",
        "expect_compat": True,
    },
    "codescout-4b": {
        "repo": "CodeScout-4B",
        "why": "Code-search/localization RL from our exact base; brings retrieval deltas",
        "license_note": "verify repo license before redistribute",
        "expect_compat": True,
    },
    "jackrong-qwen35": {
        "repo": "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled",
        "why": "Reasoning distill, BUT Qwen3.5 arch family — expect INCOMPATIBLE",
        "license_note": "quarantine (Claude-derived) + arch mismatch; LoRA-or-retrain only",
        "expect_compat": False,
    },
}

KEYS = ("architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "intermediate_size", "vocab_size", "rms_norm_eps",
        "rope_theta", "sliding_window")


def fetch_config(repo: str) -> dict:
    from huggingface_hub import hf_hub_download
    import json
    p = hf_hub_download(repo, "config.json")
    return json.loads(open(p).read())


def compat_report(base_cfg: dict, donor_cfg: dict) -> dict:
    diffs = {k: (base_cfg.get(k), donor_cfg.get(k))
             for k in KEYS if base_cfg.get(k) != donor_cfg.get(k)}
    return {"compatible": not diffs, "diffs": diffs}


def check_donor(name: str, base_cfg: dict | None = None) -> dict:
    info = DONORS[name]
    base_cfg = base_cfg or fetch_config(BASE)
    donor_cfg = fetch_config(info["repo"])
    rep = compat_report(base_cfg, donor_cfg)
    return {"donor": name, "repo": info["repo"], **rep,
            "license_note": info["license_note"], "why": info["why"]}
