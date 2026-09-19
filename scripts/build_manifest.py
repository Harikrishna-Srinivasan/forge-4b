"""Data quarantine manifest: every training file gets variant {A-permissive, B-quarantined},
source, license. B (Claude-derived) is trained+evaluated but NEVER distributed.
Usage: python3 scripts/build_manifest.py data/raw datasets/manifest.jsonl"""
from __future__ import annotations
import hashlib, json, os, sys

# permissive sources -> variant A; Claude-derived -> variant B
A_LICENSES = ("apache-2.0", "mit", "cc-by-4.0", "cc-by-sa", "bsd", "local")
B_MARKERS = ("opus", "claude", "anthropic")


def variant(source: str, license: str) -> str:
    s = (source + " " + license).lower()
    if any(m in s for m in B_MARKERS):
        return "B-quarantined"
    return "A-permissive"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n = 0
    out = open(dst, "w")
    for root, _, files in os.walk(src):
        for f in files:
            if not f.endswith((".jsonl", ".json")):
                continue
            p = os.path.join(root, f)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            lic = "unknown-do-not-redistribute"
            try:
                line = open(p).readline()
                d = json.loads(line)
                lic = d.get("license", lic)
            except Exception:
                pass
            out.write(json.dumps({"file": p, "sha": h, "license": lic,
                                  "variant": variant(p, lic)}) + "\n")
            n += 1
    print(f"manifest: {n} files -> {dst} (A trains shippable model; B stays quarantined)")


if __name__ == "__main__":
    main()
