"""kaggle_push.py — publish the FULL repo zip as a private Kaggle Dataset (versioned).
Needs: pip install kaggle; env KAGGLE_USERNAME + KAGGLE_KEY (Account > API > Create).
Run:  python3 scripts/pack.py --full && python3 scripts/kaggle_push.py --slug forge-4b-src
Then in a Kaggle notebook (GPU on, internet on): Add Input > your dataset, then:
  !unzip -q /kaggle/input/<slug>/forge-4b-full.zip -d /kaggle/working/forge-4b
  !cd /kaggle/working/forge-4b && python colab/00_setup.py
"""
from __future__ import annotations
import argparse, json, os, subprocess

FULL = "/tmp/opencode/forge-4b-full.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="dataset slug (username auto-prefixed)")
    ap.add_argument("--title", default="forge-4b-src")
    a = ap.parse_args()
    assert os.path.exists(FULL), f"run python3 scripts/pack.py --full first ({FULL} missing)"
    d = "/tmp/opencode/kaggle_ds"
    os.makedirs(d, exist_ok=True)
    import shutil
    shutil.copy(FULL, os.path.join(d, "forge-4b-full.zip"))
    import pathlib as _pl
    # username from kaggle config (works with KGAT token)
    try:
        user = json.loads(open(_pl.Path.home() / ".kaggle" / "kaggle.json").read()).get("username") or "harikrishnasriniv"
    except Exception:
        user = os.environ.get("KAGGLE_USERNAME", "harikrishnasriniv")
    meta = {"title": a.title, "id": f"{user}/{a.slug}",
            "licenses": [{"name": "apache-2.0"}]}
    open(os.path.join(d, "dataset-metadata.json"), "w").write(json.dumps(meta, indent=2))
    try:
        subprocess.check_call(["kaggle", "datasets", "create", "-p", d])
    except subprocess.CalledProcessError:
        subprocess.check_call(["kaggle", "datasets", "version", "-p", d, "-m", "update"])
    print("pushed full zip as dataset; attach it as Input in your GPU notebook")


if __name__ == "__main__":
    main()
