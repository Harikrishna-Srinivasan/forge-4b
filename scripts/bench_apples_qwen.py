#!/usr/bin/env python3
"""Fast apples Qwen-only: MMLU 5-shot, MMLU-Pro 5-shot, GSM8K 8-shot, HumanEval 0-shot — no MATH (hung)."""
import os, re, subprocess, tempfile, sys, time, random, json, math
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
def ask(m,p,npred=64):
    for _ in range(2):
        try: return chat(m, [{"role":"system","content":"You are helpful. Be concise."},{"role":"user","content":p}], temperature=0.2, num_ctx=4096, num_predict=npred, think=False)
        except Exception: time.sleep(3)
    return "[failed]"
def bench_mmlu(m, shots=5, limit=20):
    ds=load_dataset("cais/mmlu","all",split="test"); ok=0
    for i in range(min(limit,len(ds))):
        ex=ds[i]; q=ex["question"]; ch=ex["choices"]; opts="\n".join([f"{chr(65+j)}. {ch[j]}" for j in range(4)]); gold=chr(65+ex["answer"])
        few="\n\n".join([f"Q: {ds[limit+k]['question']}\n"+"\n".join([f"{chr(65+j)}. {ds[limit+k]['choices'][j]}" for j in range(4)])+f"\nAnswer: {chr(65+ds[limit+k]['answer'])}" for k in range(shots)])+"\n\n" if shots else ""
        prompt=f"{few}Q: {q}\n{opts}\nAnswer with single letter A-D only."
        mm=re.search(r"\b([A-D])\b", ask(m,prompt,npred=32)); ok+=((mm.group(1)==gold) if mm else 0); time.sleep(0.2)
    return ok, min(limit,len(ds))
def bench_mmlu_pro(m, shots=5, limit=20):
    ds=load_dataset("TIGER-Lab/MMLU-Pro",split="test"); ok=0
    for i in range(min(limit,len(ds))):
        ex=ds[i]; q=ex["question"]; opts="\n".join([f"{chr(65+j)}. {o}" for j,o in enumerate(ex["options"])]); gold=ex["answer"]
        few="\n\n".join([f"Q: {ds[limit+k]['question']}\n"+"\n".join([f"{chr(65+j)}. {ds[limit+k]['options'][j]}" for j in range(len(ds[limit+k]["options"]))])+f"\nAnswer: {ds[limit+k]['answer']}" for k in range(shots)])+"\n\n" if shots else ""
        prompt=f"{few}Q: {q}\n{opts}\nAnswer with single letter A-J only."
        mm=re.search(r"\b([A-J])\b", ask(m,prompt,npred=32)); ok+=((mm.group(1)==gold) if mm else 0); time.sleep(0.2)
    return ok, min(limit,len(ds))
def bench_gsm(m, shots=8, limit=20):
    try: ds=load_dataset("openai/gsm8k","main",split="test")
    except: ds=load_dataset("gsm8k","main",split="test")
    ok=0
    for i in range(min(limit,len(ds))):
        ex=ds[i]; gold=re.search(r"####\s*(-?\d+)",ex["answer"]); gold=gold.group(1) if gold else "?"
        def _g(idx):
            m=re.search(r"####\s*(-?\d+)", ds[idx]["answer"])
            return m.group(1) if m else "0"
        few="\n".join([f"Q: {ds[limit+k]['question']}\nA: <a>{_g(limit+k)}</a>" for k in range(shots)])+"\n" if shots else ""
        prompt=f"{few}Q: {ex['question']}\nAnswer with number only inside <a> tags."
        mm=re.search(r"<a>\s*(-?\d+)", ask(m,prompt,npred=64)); ok+=((mm.group(1)==gold) if mm else 0); time.sleep(0.2)
    return ok, min(limit,len(ds))
def bench_he(m, limit=20):
    try: ds=load_dataset("openai/openai_humaneval",split="test")
    except: ds=load_dataset("openai_humaneval",split="test")
    PRE="from typing import List, Dict, Tuple, Optional, Any\nimport math\n"; ok=0
    for i in range(min(limit,len(ds))):
        ex=ds[i]; a=ask(m,"Complete python:\n"+ex["prompt"]+"\nReturn ```python block only.",npred=512); mm=re.search(r"```python(.*?)```",a,re.S); code=(mm.group(1) if mm else a).strip()
        with tempfile.TemporaryDirectory() as d:
            p=os.path.join(d,"s.py"); open(p,"w").write(PRE+code+"\n"+ex["test"]); r=subprocess.run(["python3",p],capture_output=True,text=True,timeout=10); ok+=(r.returncode==0)
        time.sleep(0.2)
    return ok, min(limit,len(ds))
import argparse
ap=argparse.ArgumentParser(); ap.add_argument("--model",default="kamekichi128/qwen3-4b-instruct-2507:latest"); ap.add_argument("--limit",type=int,default=20); args=ap.parse_args()
M=args.model; lim=args.limit
print(f"=== APPLES-QWEN model={M} ===",flush=True)
for name,fn in [("MMLU 5-shot", lambda: bench_mmlu(M,5,lim)), ("MMLU-Pro 5-shot", lambda: bench_mmlu_pro(M,5,lim)), ("GSM8K 8-shot", lambda: bench_gsm(M,8,lim)), ("HumanEval 0-shot", lambda: bench_he(M,lim))]:
    try: o,t=fn(); print(f"{name}: {o}/{t} {o/t*100:.1f}%",flush=True)
    except Exception as e: print(f"{name}: FAILED {e}",flush=True)
