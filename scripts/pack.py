"""pack.py — zip code+datasets for Colab/Kaggle upload (excludes models/, __pycache__).
Run: python3 scripts/pack.py  ->  /tmp/opencode/forge-4b-pack.zip
Upload that + run colab/00_setup.py first."""
from __future__ import annotations
import os, zipfile

ROOT = "/home/dell/forge-4b"
OUT = "/tmp/opencode/forge-4b-pack.zip"
FULL_OUT = "/tmp/opencode/forge-4b-full.zip"
SKIP = ("models", "__pycache__", ".git", "artifacts")


def main():
    import sys
    full = "--full" in sys.argv
    out = FULL_OUT if full else OUT
    skip = ("__pycache__", ".git") if full else SKIP
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in skip]
            for f in files:
                if f.endswith(".pyc"):
                    continue
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, ROOT))
                n += 1
    print(f"pack({'full' if full else 'slim'}): {n} files -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
