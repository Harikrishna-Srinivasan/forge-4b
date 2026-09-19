"""Persistent resumable task state. Every artifact versioned, every example keeps provenance."""
from __future__ import annotations
import json, os, time, uuid

ART = os.environ.get("FORGE_ARTIFACTS", "/home/dell/forge-4b/artifacts")

class Store:
    def __init__(self, root: str = ART):
        self.root = root; os.makedirs(root, exist_ok=True)
    def save(self, kind: str, obj: dict) -> str:
        vid = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.json"
        p = os.path.join(self.root, vid)
        open(p, "w").write(json.dumps(obj, indent=2)[:200000])
        return p
    def append_jsonl(self, name: str, obj: dict):
        p = os.path.join(self.root, name)
        open(p, "a").write(json.dumps(obj)[:20000] + "\n")
        return p
