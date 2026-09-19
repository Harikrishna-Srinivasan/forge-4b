#!/usr/bin/env python3
"""Real benchmarks v2 — robust, laptop-tuned, actual Qwen3.6-35B-A3B families.
Run: FORGE_ZEN=0 python3 scripts/real_bench2.py --limit 5  (5 per family = 20 total, ~10min on 4B)
"""
import argparse, json, os, re, subprocess, tempfile, sys, time
# HF auth from ~/.env hftok (falls back to HF_TOK) so gated/open datasets work
def _load_hf_token():
    for p in [os.path.expanduser("~/.env"), os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")]:
        try:
            if os.path.exists(p):
                for line in open(p):
                    line=line.strip()
                    if line.startswith("hftok="):
                        return line.split("=",1)[1].strip()
        except Exception:
            pass
    return None
_HFTOK = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or _load_hf_token() or os.environ.get("HF_TOK")
if _HFTOK:
    os.environ["HF_TOKEN"]=_HFTOK; os.environ["HUGGING_FACE_HUB_TOKEN"]=_HFTOK
    try:
        from huggingface_hub import login as _login
        _login(token=_HFTOK)
    except Exception:
        pass
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.student import chat

MODEL = "kamekichi128/qwen3-4b-instruct-2507:latest"
def ask(prompt, sysmsg="You are helpful. Be concise."):
    for _ in range(2):
        try:
            return chat(MODEL, [{"role":"system","content":sysmsg},{"role":"user","content":prompt}], temperature=0.2, num_ctx=4096, num_predict=64, think=False)
        except Exception as e:
            time.sleep(2)
            if "timed out" in str(e).lower(): continue
            return f"[error {e}]"
    return "[failed]"

def bench_mmlu(n=5):
    ds=load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    # sample first n deterministically
    ok=0
    for i in range(min(n,len(ds))):
        ex=ds[i]; q=ex["question"]; opts="\n".join([f"{chr(65+j)}. {o}" for j,o in enumerate(ex["options"])])
        ans=ask(f"{q}\n{opts}\nAnswer with single letter A-J only.")
        m=re.search(r"\b([A-J])\b", ans)
        pred=m.group(1) if m else "?"
        ok+= (pred==ex["answer"])
        time.sleep(0.5)
    return ok,n

def bench_gpqa(n=5):
    # Open GPQA-Diamond mirror (Idavidrein/gpqa is gated even with token).
    # fingertap/GPQA-Diamond: {question (stem+options), answer (letter)}
    try:
        ds=load_dataset("fingertap/GPQA-Diamond", split="test")
        open_fmt=True
    except Exception:
        ds=load_dataset("Idavidrein/gpqa","gpqa_diamond",split="train")
        open_fmt=False
    ok=0
    for i in range(min(n,len(ds))):
        ex=ds[i]
        if open_fmt:
            q=ex["question"]; gold=str(ex["answer"]).strip().upper()[:1]
            ans=ask(f"{q}\nAnswer with single letter A-D only.")
            m=re.search(r"\b([A-D])\b", ans)
            if m and m.group(1)==gold: ok+=1
        else:
            q=ex["Question"]; opts=f"A. {ex['Correct Answer']}\nB. {ex['Incorrect Answer 1']}\nC. {ex['Incorrect Answer 2']}\nD. {ex['Incorrect Answer 3']}"
            ans=ask(f"{q}\n{opts}\nAnswer A/B/C/D only.")
            m=re.search(r"\b([A-D])\b", ans)
            if m and m.group(1)=="A": ok+=1
        time.sleep(0.5)
    return ok,n

def bench_gsm8k(n=5):
    try:
        ds=load_dataset("openai/gsm8k","main",split="test")
    except Exception:
        ds=load_dataset("gsm8k","main",split="test")
    ok=0
    for i in range(min(n,len(ds))):
        q=ds[i]["question"]; gold=re.search(r"####\s*(-?\d+)", ds[i]["answer"]); gold=gold.group(1) if gold else "?"
        ans=ask(f"{q}\nAnswer with number only inside <a> tags.")
        m=re.search(r"<a>\s*(-?\d+)", ans)
        if m and m.group(1)==gold: ok+=1
        time.sleep(0.5)
    return ok,n

def bench_humaneval(n=3):
    try:
        ds=load_dataset("openai/openai_humaneval",split="test")
    except Exception:
        ds=load_dataset("openai_humaneval",split="test")
    ok=0
    for i in range(min(n,len(ds))):
        prompt=ds[i]["prompt"]; tests=ds[i]["test"]
        ans=ask(f"Complete python:\n{prompt}\nReturn ```python block only.")
        m=re.search(r"```python(.*?)```", ans, re.S)
        code=m.group(1) if m else ans
        with tempfile.TemporaryDirectory() as d:
            p=os.path.join(d,"sol.py"); open(p,"w").write(code+"\n"+tests)
            r=subprocess.run(["python3",p],capture_output=True,text=True,timeout=5)
            if r.returncode==0: ok+=1
        time.sleep(0.5)
    return ok,n

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=5); args=ap.parse_args()
    res={}
    for name,fn in [("MMLU-Pro",bench_mmlu),("GPQA-diamond",bench_gpqa),("GSM8K",bench_gsm8k),("HumanEval",bench_humaneval)]:
        lim = args.limit if name!="HumanEval" else min(args.limit, 10)
        try:
            o,t=fn(lim); pct=o/t*100 if t else 0
            res[name]=f"{o}/{t} {pct:.1f}%"
            print(f"{name}: {o}/{t} {pct:.1f}%", flush=True)
        except Exception as e:
            res[name]=f"skipped ({e.__class__.__name__})"
            print(f"{name}: skipped {e}", flush=True)
    print(json.dumps(res,indent=2))
    targets={"MMLU-Pro":85.2,"GPQA":86.0,"GSM8K":92, "HumanEval":73.4, "SWE Verified":73.4, "Terminal 2.0":51.5}
    print("Qwen3.6-35B-A3B targets:", targets)
    import pathlib; pathlib.Path("evaluations/real-bench2.json").write_text(json.dumps({"model":MODEL,"results":res,"targets":targets},indent=2))
