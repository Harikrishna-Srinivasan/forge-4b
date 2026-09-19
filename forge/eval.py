"""Tiny offline eval suites (no downloads). Separate retrieval/localization/diagnosis/patch/verification."""
from __future__ import annotations
from .orchestrator import Orchestrator

SUITES = {
 "instruction": [{"problem": "Write exactly 3 bullet lines about git, each <=10 words, include 'commit', exclude 'push'.",
                  "capability": "01_instruction",
                  "constraints": {"include": ["commit"], "exclude": ["push"]}}],
 "reasoning": [{"problem": "If a*3+2=20, what is a? Answer with number only inside <a> tags./no_think",
                "capability": "02_reasoning"}],
 "coding": [{"problem": "Write python function add(a,b) returning sum. Put in ```python block./no_think",
             "capability": "04_coding", "tests": "\nassert add(2,3)==5\nassert add(-1,1)==0\n"}],
 "debugging": [{"problem": "Fix: def div(a,b): return a/b fails on b=0. Return ```python with guard./no_think",
                "capability": "05_debugging", "tests": "\nassert div(4,2)==2\ntry:\n div(1,0); assert False\n except (ZeroDivisionError, ValueError): pass\n"}],
 "contradiction": [{"problem": "Text: 'All birds fly. Penguins are birds that never fly.' Is there a contradiction? Explain in 2 sentences./no_think",
                    "capability": "13_contradiction_detection"}],
}

def run_suite(model: str = "kamekichi128/qwen3-4b-instruct-2507:latest", suite: str = "all") -> dict:
    orch = Orchestrator(model)
    names = list(SUITES) if suite == "all" else [suite]
    results = {}
    for n in names:
        ok = 0; details = []
        for t in SUITES[n]:
            r = orch.run(t["problem"], t.get("capability", "16_general_instruction"),
                         t.get("tests", ""), t.get("constraints"))
            v = r.get("verification", {})
            passed = bool(v.get("ok", False))
            # instruction suite: enforce constraint check strictly
            if n == "instruction" and not passed: passed = False
            ok += passed; details.append({"problem": t["problem"][:60], "ok": passed})
        results[n] = {"pass": ok, "total": len(SUITES[n]), "details": details}
    return results
