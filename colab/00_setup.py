"""00_setup.py — Colab/Kaggle first cell. Installs train stack, mounts workspace.
Run: !python colab/00_setup.py [--hf-token $HF_TOKEN]
Needs: GPU runtime (T4+ minimum; A100 for GRPO). Laptop never runs this."""
from __future__ import annotations
import subprocess, sys

PKGS = ["unsloth[colab-newest]", "trl", "peft", "datasets", "huggingface_hub",
        "safetensors", "bitsandbytes"]


def main():
    for p in PKGS:
        print(f"== install {p}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
    import torch
    print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"vram: {free/1e9:.1f}/{total/1e9:.1f} GB free")
    except Exception as e:
        print("vram-query-fail:", e)


if __name__ == "__main__":
    main()
