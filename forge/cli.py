"""forge CLI: forge run --stage all and individual stages. Resumable, versioned artifacts.
train-* stages EXPORT data for remote GPU (no local fine-tune per constraint)."""
from __future__ import annotations
import argparse, json, sys
from .eval import run_suite
from .orchestrator import Orchestrator
from .state import Store

def main():
    ap = argparse.ArgumentParser(prog="forge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--stage", default="all")
    r.add_argument("--model", default="qwen3:4b"); r.add_argument("--problem", default="")
    # alias: `forge ask "question"` == `forge run --stage generate --problem "question"`
    ask_p = sub.add_parser("ask", help="single question via harness (verifier+tools)")
    ask_p.add_argument("problem", nargs="?", default="")
    ask_p.add_argument("--problem", dest="ask_problem", default="")
    ask_p.add_argument("--model", default="qwen3:4b")
    ask_p.add_argument("--track", default="A", choices=["A","B","C"], help="arena track (A/B/C)")
    ask_p.add_argument("--capability", default="04_coding",
                       choices=["01_instruction", "02_reasoning", "04_coding", "05_debugging"],
                       help="select verifier/prompt behavior")
    ask_p.add_argument("--tests", default="", help="Python tests appended to generated code")
    for s in ["research","generate","verify","filter","train-sft","train-dpo","train-rl",
              "evaluate","analyze","improve"]:
        p = sub.add_parser(s); p.add_argument("--model", default="qwen3:4b")
    a = ap.parse_args()
    store = Store()
    if a.cmd == "ask":
        prob = (getattr(a, "ask_problem", "") or getattr(a, "problem", "")).strip()
        if not prob:
            ap.error("ask requires a question: forge ask \"your question\" or forge ask --problem \"...\"")
        orch = Orchestrator(a.model)
        # Do not claim verification without a task-specific test suite.
        out = orch.run(prob, a.capability, a.tests)
        print(json.dumps({k: str(v)[:2000] for k, v in out.items()}, indent=2))
        print("saved:", store.save("task", out))
        return
    if a.cmd == "run":
        stage = a.stage
        if stage in ("all", "evaluate"):
            res = run_suite(a.model, "all")
            print(json.dumps(res, indent=2))
            print("saved:", store.save("eval", res))
        elif stage in ("generate", "research"):
            orch = Orchestrator(a.model)
            prob = a.problem or "Write python function mul(a,b). ```python block./no_think"
            out = orch.run(prob, "04_coding", "\nassert mul(2,3)==6\n")
            print(json.dumps({k: str(v)[:1500] for k, v in out.items()}, indent=2))
            print("saved:", store.save("task", out))
            store.append_jsonl("dataset.jsonl", {"problem": out["problem"], "answer": out["answer"][:2000],
                "verification": out["verification"], "model": a.model, "license": "Apache-2.0(local)"})
        else:
            print(json.dumps({"stage": stage, "note": "see individual subcommand"}))
        return
    if a.cmd == "evaluate":
        res = run_suite(a.model, "all"); print(json.dumps(res, indent=2))
        print("saved:", store.save("eval", res)); return
    if a.cmd in ("train-sft", "train-dpo", "train-rl"):
        print(json.dumps({
         "stage": a.cmd, "mode": "EXPORT-ONLY (no local fine-tune)",
         "dataset": "/home/dell/forge-4b/artifacts/dataset.jsonl",
         "remote_recipe": "Use rented A100 80GB: unsloth QLoRA Qwen3-4B-Instruct-2507, 5k-400k trajs, then DPO/GRPO with verifiable rewards.",
         "reason": "laptop has no GPU; laptop = orchestration/eval/small inference only"}, indent=2))
        return
    print(json.dumps({"cmd": a.cmd, "status": "ok",
                      "hint": "forge run --stage [all|generate|evaluate] works offline with ollama qwen3:4b"}))

if __name__ == "__main__":
    main()
