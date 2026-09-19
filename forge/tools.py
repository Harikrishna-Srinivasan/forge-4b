"""Sandboxed terminal tools. Authorized isolated env only. No network, no destructive cmds."""
from __future__ import annotations
import os, subprocess, pathlib

ROOT = os.environ.get("FORGE_SANDBOX", "/tmp/opencode")
BLOCKED = ("rm -rf /", "mkfs", ":(){", "dd if=", "shutdown", "reboot", "curl ", "wget ")

def _guard(cmd: str):
    for b in BLOCKED:
        if b in cmd:
            raise ValueError(f"blocked: {b.strip()}")

def run(cmd: str, cwd: str = "", timeout: int = 15) -> str:
    _guard(cmd)
    base = pathlib.Path(ROOT); base.mkdir(parents=True, exist_ok=True)
    c = (cwd or str(base))
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=c)
    return (r.stdout + r.stderr)[-4000:]

def ls(path: str = ".") -> str: return run(f"ls -la {path}")
def find_grep(pattern: str, path: str = ".") -> str: return run(f"grep -rn --include='*.py' {pattern} {path} | head -30")
