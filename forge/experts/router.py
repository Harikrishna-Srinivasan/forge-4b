"""LoRA-expert router skeleton (total ≤4.5B: frozen 4B base + one active ~0.1B LoRA).

v1: deterministic capability-tag routing (no learned weights, no misroute training).
v2 (only if v1 misroute >10% on arena): tiny classifier trained on arena prompts.
Remote (Colab) trains the LoRAs; laptop only routes + evaluates merged copies.
"""
from __future__ import annotations
import os

EXPERTS = ("code", "debug-localize", "terminal-tools", "reason-cot")

TAG_MAP = {
    "04_coding": "code", "05_debugging": "debug-localize",
    "06_code_localization": "debug-localize", "07_repository_reasoning": "debug-localize",
    "08_terminal": "terminal-tools", "09_tool_use": "terminal-tools",
    "02_reasoning": "reason-cot", "03_math": "reason-cot",
    "13_contradiction_detection": "reason-cot",
}

# Aspect names used by arena50 -> expert
ASPECT_MAP = {
    "coding": "code", "debugging": "code", "longhorizon_repo": "debug-localize",
    "terminal_react": "terminal-tools", "reasoning_CoT": "reason-cot",
    "reasoning_ToT": "reason-cot", "instruction": "code",
    "contradiction_gaslight": "reason-cot", "security_defensive": "debug-localize",
    "creativity_generalize": "code",
}


def route(capability_or_aspect: str) -> str:
    return TAG_MAP.get(capability_or_aspect,
                       ASPECT_MAP.get(capability_or_aspect, "code"))


def active_model_dir(models_root: str = "/home/dell/forge-4b/models") -> dict:
    """Which merged/adapter dir to load per expert (populated by your GPU runs)."""
    return {e: os.path.join(models_root, f"forge-4b-{e}") for e in EXPERTS}
