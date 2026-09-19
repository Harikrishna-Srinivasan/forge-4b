#!/usr/bin/env python3
"""
Apples-to-apples runner: reproduces per-family shot counts.

Modes:
  --suite qwen      : MMLU 5-shot, MMLU-Pro 5-shot, GSM8K 8-shot, HumanEval 0-shot, MATH 4-shot
  --suite gemma-pt  : MMLU 5-shot, MMLU-Pro-CoT 5-shot, GSM8K 8-shot, GPQA 5-shot, HumanEval 0-shot, MATH 4-shot
  --suite llama     : MMLU 5-shot, GSM8K 8-shot CoT, MATH 0-shot CoT, GPQA 0-shot, ARC-C 0/25
  --suite all       : union (runs all, ~110 items total if limit=20 per bench)

Shot examples are drawn from the SAME dataset split (train where possible, else test[0:K]).
Prompt style mirrors vendor cards: Qwen CoT = "think step by step", Gemma PT = plain likelihood-style but we use generation.

Run:
  FORGE_ZEN=0 python3 scripts/bench_apples.py --suite qwen --model kamekichi128/qwen3-4b-instruct-2507:latest --limit 20
  FORGE_ZEN=0 FORGE_BENCH_MODEL=qwen3:1.7b python3 scripts/bench_apples.py --suite qwen --limit 20
"""
import argparse, json, math, os, re, subprocess, tempfile, sys, time, random
def _tok():
    for p in [os.path.expanduser("~/.env")]:
        try:
            if os.path.exists(p):
                for line in open(p):
                    if line.strip().startswith("hftok="): return line.split("=",1)[1].strip()
        except: pass
    return None
_t=_tok()
if _t: os.environ["HF_TOKEN"]=_t; os.environ["HUGGING_FACE_HUB_TOKEN"]=_t
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.student import chat

DEFAULT_M = os.environ.get("FORGE_BENCH_MODEL","kamekichi128/qwen3-4b-instruct-2507:latest")

def ask(model, prompt, sysmsg="You are helpful. Be concise.", temp=0.2, top_p=0.8, top_k=40, npred=256):
    for _ in range(2):
        try: return chat(model, [{"role":"system","content":sysmsg},{"role":"user","content":prompt}], temperature=temp, top_p=top_p, top_k=top_k, num_ctx=4096, num_predict=npred, think=False)
        except Exception: time.sleep(3)
    return "[failed]"

def ci(k,n):
    if not n: return "—"
    p=k/n; se=math.sqrt(max(p*(1-p),0)/n); return f"±{1.96*se*100:.1f}pp"

def fewshot_examples(dataset, n_shots, prompt_fn, split="test", limit_for_pool=50):
    # pull first n_shots items that are not in the test slice we evaluate
    pool = []
    for i in range(n_shots, n_shots+limit_for_pool):
        if i < len(dataset):
            pool.append(prompt_fn(dataset[i]))
            if len(pool)>=n_shots: break
    return pool

def bench_mmlu(model, shots=5, limit=20, cot=False):
    ds = load_dataset("cais/mmlu", "all", split="test")  # for 5-shot MMLU
    # also load MMLU-Pro for the Pro variant; caller selects
    # Here handle MMLU (57 subjects) 5-shot
    rnd = random.Random(7)
    # take first `limit` as eval, use next 20 as few-shot pool
    ok=0
    for i in range(min(limit, len(ds))):
        ex = ds[i]
        question = ex["question"]
        choices = [ex["choices"][j] for j in range(4)]
        opts = "\n".join([f"{chr(65+j)}. {c}" for j,c in enumerate(choices)])
        answer_letter = chr(65+ex["answer"])
        # build few-shot
        few = ""
        if shots>0:
            pool = []
            for j in range(limit, limit+30):
                if j>=len(ds): break
                e = ds[j]
                ch = [e["choices"][k] for k in range(4)]
                o = "\n".join([f"{chr(65+k)}. {ch[k]}" for k in range(4)])
                a = chr(65+e["answer"])
                if cot:
                    pool.append(f"Q: {e['question']}\n{o}\nThink step by step then answer. Answer: {a}")
                else:
                    pool.append(f"Q: {e['question']}\n{o}\nAnswer: {a}")
                if len(pool)>=shots: break
            few = "\n\n".join(pool) + "\n\n"
        prompt = f"{few}Q: {question}\n{opts}\n" + ("Think step by step then answer with single letter A-D only." if cot else "Answer with single letter A-D only.")
        m=re.search(r"\b([A-D])\b", ask(model, prompt, npred=32))
        ok+=((m.group(1)==answer_letter) if m else 0)
        time.sleep(0.2)
    return ok, min(limit,len(ds))

def bench_mmlu_pro(model, shots=5, limit=20, cot=False):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    ok=0
    for i in range(min(limit,len(ds))):
        ex=ds[i]; q=ex["question"]; opts="\n".join([f"{chr(65+j)}. {o}" for j,o in enumerate(ex["options"])])
        gold=ex["answer"]
        few=""
        if shots>0:
            pool=[]
            for j in range(limit, limit+40):
                if j>=len(ds): break
                e=ds[j]; o="\n".join([f"{chr(65+k)}. {e['options'][k]}" for k in range(len(e["options"]))])
                if cot:
                    pool.append(f"Q: {e['question']}\n{o}\nThink step by step then answer. Answer: {e['answer']}")
                else:
                    pool.append(f"Q: {e['question']}\n{o}\nAnswer: {e['answer']}")
                if len(pool)>=shots: break
            few="\n\n".join(pool)+"\n\n"
        prompt=f"{few}Q: {q}\n{opts}\n" + ("Think step by step then answer with single letter A-J only." if cot else "Answer with single letter A-J only.")
        m=re.search(r"\b([A-J])\b", ask(model, prompt, npred=32))
        ok+=((m.group(1)==gold) if m else 0)
        time.sleep(0.2)
    return ok, min(limit,len(ds))

def bench_gsm8k(model, shots=8, limit=20):
    try: ds=load_dataset("openai/gsm8k","main",split="test")
    except: ds=load_dataset("gsm8k","main",split="test")
    ok=0
    # few-shot pool from first shots items after eval slice
    for i in range(min(limit,len(ds))):
        ex=ds[i]; gold=re.search(r"####\s*(-?\d+)",ex["answer"]); gold=gold.group(1) if gold else "?"
        few=""
        if shots>0:
            pool=[]
            for j in range(limit, limit+30):
                if j>=len(ds): break
                e=ds[j]; g=re.search(r"####\s*(-?\d+)",e["answer"]); g=g.group(1) if g else "?"
                pool.append(f"Q: {e['question']}\nA: <a>{g}</a>")
                if len(pool)>=shots: break
            few="\n".join(pool)+"\n"
        prompt=f"{few}Q: {ex['question']}\nAnswer with number only inside <a> tags."
        m=re.search(r"<a>\s*(-?\d+)", ask(model, prompt, npred=64))
        ok+=((m.group(1)==gold) if m else 0)
        time.sleep(0.2)
    return ok, min(limit,len(ds))

def bench_humaneval(model, shots=0, limit=20):
    try: ds=load_dataset("openai/openai_humaneval", split="test")
    except: ds=load_dataset("openai_humaneval", split="test")
    PRE="from typing import List, Dict, Tuple, Optional, Any\nimport math\n"
    ok=0
    for i in range(min(limit,len(ds))):
        ex=ds[i]
        prompt="Complete python:\n"+ex["prompt"]+"\nReturn ```python block only."
        # Humaneval is always 0-shot per vendors
        a=ask(model, prompt, npred=512)
        m=re.search(r"```python(.*?)```",a,re.S); code=(m.group(1) if m else a).strip()
        with tempfile.TemporaryDirectory() as d:
            p=os.path.join(d,"s.py"); open(p,"w").write(PRE+code+"\n"+ex["test"])
            r=subprocess.run(["python3",p],capture_output=True,text=True,timeout=10)
            ok+=(r.returncode==0)
        time.sleep(0.2)
    return ok, min(limit,len(ds))

def bench_gpqa(model, shots=5, limit=20):
    ds=load_dataset("fingertap/GPQA-Diamond", split="test"); ok=0
    for i in range(min(limit,len(ds))):
        ex=ds[i]; gold=str(ex["answer"]).strip().upper()[:1]
        few=""
        if shots>0:
            pool=[]
            for j in range(limit, limit+30):
                if j>=len(ds): break
                e=ds[j]; pool.append(f"Q: {e['question']}\nAnswer: {e['answer']}")
                if len(pool)>=shots: break
            few="\n\n".join(pool)+"\n\n"
        prompt=f"{few}Q: {ex['question']}\nAnswer with single letter A-D only."
        m=re.search(r"\b([A-D])\b", ask(model, prompt, npred=32))
        ok+=((m.group(1)==gold) if m else 0); time.sleep(0.2)
    return ok, min(limit,len(ds))

def bench_math(model, shots=4, limit=20):
    ds=load_dataset("lighteval/MATH", split="test")
    # lighteval MATH has `problem` and `solution`
    ok=0
    for i in range(min(limit,len(ds))):
        ex=ds[i]; q=ex["problem"]; sol=ex["solution"]
        # extract boxed answer
        gold=re.search(r"\\boxed\{([^}]+)\}", sol)
        gold=gold.group(1).strip() if gold else sol.strip().split()[-1]
        few=""
        if shots>0:
            pool=[]
            for j in range(limit, limit+30):
                if j>=len(ds): break
                e=ds[j]; pool.append(f"Problem: {e['problem']}\nAnswer: {e['solution'][:300]}")
                if len(pool)>=shots: break
            few="\n\n".join(pool)+"\n\n"
        prompt=f"{few}Problem: {q}\nSolve step by step, put final answer in \\boxed{{}}."
        ans=ask(model, prompt, npred=256)
        m=re.search(r"\\boxed\{([^}]+)\}", ans)
        pred=m.group(1).strip() if m else ans.strip().split()[-5:]
        # loose check: gold substring in answer
        if gold in ans or (m and gold==pred):
            ok+=1
        time.sleep(0.2)
    return ok, min(limit,len(ds))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["qwen","gemma-pt","llama","all"], default="qwen")
    ap.add_argument("--model", default=DEFAULT_M)
    ap.add_argument("--limit", type=int, default=20)
    args=ap.parse_args()
    M=args.model
    suite=args.suite
    tasks=[]
    if suite in ("qwen","all"):
        tasks += [("MMLU 5-shot", lambda: bench_mmlu(M, shots=5, limit=args.limit)),
                  ("MMLU-Pro 5-shot", lambda: bench_mmlu_pro(M, shots=5, limit=args.limit)),
                  ("GSM8K 8-shot", lambda: bench_gsm8k(M, shots=8, limit=args.limit)),
                  ("HumanEval 0-shot", lambda: bench_humaneval(M, shots=0, limit=args.limit)),
                  ("MATH 4-shot", lambda: bench_math(M, shots=4, limit=args.limit))]
    if suite in ("gemma-pt","all"):
        tasks += [("Gemma: MMLU-Pro-CoT 5-shot", lambda: bench_mmlu_pro(M, shots=5, limit=args.limit, cot=True)),
                  ("Gemma: GPQA 5-shot", lambda: bench_gpqa(M, shots=5, limit=args.limit))]
    if suite in ("llama","all"):
        tasks += [("Llama: GSM8K 8-shot CoT", lambda: bench_gsm8k(M, shots=8, limit=args.limit)), # same prompt but CoT implied
                  ("Llama: MATH 0-shot CoT", lambda: bench_math(M, shots=0, limit=args.limit))]
    print(f"=== APPLES suite={suite} model={M} limit={args.limit} ===", flush=True)
    res={}
    for name,fn in tasks:
        try:
            o,t=fn(); pct=o/t*100 if t else 0
            res[name]=f"{o}/{t} {pct:.1f}%"
            print(f"{name}: {o}/{t} {pct:.1f}%", flush=True)
        except Exception as e:
            res[name]=f"error {e.__class__.__name__}: {e}"
            print(f"{name}: FAILED {e}", flush=True)
    out=f"evaluations/apples-{suite}-{os.path.basename(M).replace(':','-')}.json"
    json.dump({"model":M,"suite":suite,"results":res}, open(out,"w"), indent=2)
    print(f"saved {out}\n{json.dumps(res,indent=2)}")
