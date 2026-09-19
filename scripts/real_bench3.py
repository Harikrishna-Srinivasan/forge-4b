#!/usr/bin/env python3
"""V1 inference-peak: maj@3 self-consistency (knowledge/math) + exec-retry K=2 (code).
Same items as real_bench2 (first-n deterministic) so subsets compare directly.
Run: FORGE_ZEN=0 python3 scripts/real_bench3.py --limit 10
"""
import argparse, json, os, re, subprocess, tempfile, sys, time, collections
def _load_hf_token():
    for p in [os.path.expanduser("~/.env")]:
        try:
            if os.path.exists(p):
                for line in open(p):
                    line=line.strip()
                    if line.startswith("hftok="):
                        return line.split("=",1)[1].strip()
        except Exception:
            pass
    return None
_HFTOK = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or _load_hf_token()
if _HFTOK:
    os.environ["HF_TOKEN"]=_HFTOK; os.environ["HUGGING_FACE_HUB_TOKEN"]=_HFTOK
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.student import chat

MODEL = "kamekichi128/qwen3-4b-instruct-2507:latest"
def ask(prompt, sysmsg="You are helpful. Be concise.", temp=0.7):
    try:
        return chat(MODEL, [{"role":"system","content":sysmsg},{"role":"user","content":prompt}], temperature=temp, num_ctx=4096, num_predict=64, think=False)
    except Exception as e:
        time.sleep(2)
        try:
            return chat(MODEL, [{"role":"system","content":sysmsg},{"role":"user","content":prompt}], temperature=temp, num_ctx=4096, num_predict=64, think=False)
        except Exception as e2:
            return f"[error {e2}]"

def maj_letter(prompt, letters="A-J", k=3):
    votes=[]
    for t in [0.7, 0.9, 0.5][:k]:
        ans=ask(prompt, temp=t); time.sleep(0.3)
        m=re.search(r"\b([%s])\b" % letters, ans)
        if m: votes.append(m.group(1))
    if not votes: return "?"
    c=collections.Counter(votes)
    return c.most_common(1)[0][0]

def bench_mmlu(n=5):
    ds=load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ok=0
    for i in range(min(n,len(ds))):
        ex=ds[i]; q=ex["question"]; opts="\n".join([f"{chr(65+j)}. {o}" for j,o in enumerate(ex["options"])])
        pred=maj_letter(f"{q}\n{opts}\nAnswer with single letter A-J only.", "A-J", 3)
        ok+=(pred==ex["answer"]); time.sleep(0.3)
    return ok,n

def bench_gpqa(n=5):
    ds=load_dataset("fingertap/GPQA-Diamond", split="test")
    ok=0
    for i in range(min(n,len(ds))):
        ex=ds[i]; q=ex["question"]; gold=str(ex["answer"]).strip().upper()[:1]
        pred=maj_letter(f"{q}\nAnswer with single letter A-D only.", "A-D", 3)
        ok+=(pred==gold); time.sleep(0.3)
    return ok,n

def bench_gsm8k(n=5):
    try: ds=load_dataset("openai/gsm8k","main",split="test")
    except Exception: ds=load_dataset("gsm8k","main",split="test")
    ok=0
    for i in range(min(n,len(ds))):
        q=ds[i]["question"]; gold=re.search(r"####\s*(-?\d+)", ds[i]["answer"]); gold=gold.group(1) if gold else "?"
        votes=[]
        for t in [0.7,0.9,0.5]:
            ans=ask(f"{q}\nAnswer with number only inside <a> tags.", temp=t); time.sleep(0.3)
            m=re.search(r"<a>\s*(-?\d+)", ans)
            if m: votes.append(m.group(1))
        if votes and collections.Counter(votes).most_common(1)[0][0]==gold: ok+=1
        time.sleep(0.3)
    return ok,n

def run_code(code, tests, timeout=5):
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"sol.py"); open(p,"w").write(code+"\n"+tests)
        r=subprocess.run(["python3",p],capture_output=True,text=True,timeout=timeout)
        return r.returncode==0, (r.stderr or r.stdout)[-600:]

def bench_humaneval(n=3):
    try: ds=load_dataset("openai/openai_humaneval",split="test")
    except Exception: ds=load_dataset("openai_humaneval",split="test")
    ok=0
    for i in range(min(n,len(ds))):
        prompt=ds[i]["prompt"]; tests=ds[i]["test"]
        ans=ask(f"Complete python:\n{prompt}\nReturn ```python block only.", temp=0.5)
        m=re.search(r"```python(.*?)```", ans, re.S)
        code=(m.group(1) if m else ans).strip()
        passed,_=run_code(code, tests)
        if not passed:
            # retry once with error feedback (SWE-smith style)
            ans2=ask(f"Complete python:\n{prompt}\nPrevious attempt failed tests. Fix it.\nFailed code:\n{code[:1500]}\nReturn ```python block only.", temp=0.7)
            m2=re.search(r"```python(.*?)```", ans2, re.S)
            code2=(m2.group(1) if m2 else ans2).strip()
            passed,_=run_code(code2, tests)
        if passed: ok+=1
        time.sleep(0.3)
    return ok,n

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=10); args=ap.parse_args()
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
    import pathlib; pathlib.Path("evaluations/real-bench3.json").write_text(json.dumps({"model":MODEL,"arch":"V1-maj3+execretry","results":res},indent=2))
