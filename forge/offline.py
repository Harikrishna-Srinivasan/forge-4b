"""Offline test gate: test-time runs LOCAL ONLY. Cloud teachers are train-time only.
Any evaluation with FORGE_ZEN=1 is invalid for parity claims."""
from __future__ import annotations
import os


def assert_offline() -> None:
    if os.environ.get("FORGE_ZEN", "0") == "1":
        raise SystemExit("OFFLINE-GATE: FORGE_ZEN=1 at test time is forbidden. "
                         "Cloud teachers are train-time only. Rerun with FORGE_ZEN=0.")
    for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "GLM_API_KEY"):
        if os.environ.get(k):
            raise SystemExit(f"OFFLINE-GATE: {k} set at test time. Unset it for valid runs.")
