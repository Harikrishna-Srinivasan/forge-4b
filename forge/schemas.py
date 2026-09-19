"""Data schema with provenance/license tracking. Every example keeps teacher/source metadata."""
from __future__ import annotations
import hashlib, time, uuid
from dataclasses import dataclass, field, asdict
from typing import Any

CAPABILITIES = [
 "01_instruction","02_reasoning","03_math","04_coding","05_debugging",
 "06_code_localization","07_repository_reasoning","08_terminal","09_tool_use",
 "10_web_research","11_long_context","12_pattern_connection",
 "13_contradiction_detection","14_agent_recovery","15_security_defensive",
 "16_general_instruction",
]

@dataclass
class Provenance:
    teacher: str = ""          # e.g. "qwen3:4b-sample-0", "deepseek-r1", "human"
    teacher_size: str = ""     # e.g. "4B", "32B", "unknown"
    source_dataset: str = ""   # e.g. "SWE-smith", "synthetic-forge"
    license: str = ""          # e.g. "Apache-2.0", "MIT", "unknown-research-only"
    url: str = ""
    created_ts: float = field(default_factory=time.time)
    version: str = "v0.1.0"

@dataclass
class TrajectoryStep:
    observation: str = ""
    hypothesis: str = ""
    evidence: str = ""
    action: str = ""
    result: str = ""
    updated_belief: str = ""
    verification: str = ""

@dataclass
class Example:
    example_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    capability: str = "16_general_instruction"
    problem: str = ""
    candidates: list = field(default_factory=list)  # [{agent, answer, provenance}]
    evidence: str = ""
    verification: dict = field(default_factory=dict)
    conclusion: str = ""
    trajectory: list = field(default_factory=list)  # list[dict TrajectoryStep]
    failed_then_fixed: bool = False
    provenance: Provenance = field(default_factory=Provenance)

    def uid(self) -> str:
        h = hashlib.sha256((self.problem + self.conclusion).encode()).hexdigest()[:16]
        return f"{self.capability}-{h}"

def new_provenance(teacher: str, license: str = "Apache-2.0", **kw: Any) -> Provenance:
    return Provenance(teacher=teacher, license=license, **kw)
