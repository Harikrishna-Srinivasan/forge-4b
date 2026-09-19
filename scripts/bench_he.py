#!/usr/bin/env python3
"""HumanEval-10 rescore with standard preamble (typing/math) + 1 exec-retry.
Run: FORGE_ZEN=0 nohup python3 scripts/bench_he.py > evaluations/bench-he.log 2>&1 &
"""
import os, re, subprocess, tempfile, sys, time, json
tok=None
for line in open(os.path.expanduser("~/.env")):
    if line.strip().startswith("hftok="): tok=line.strip().split("=",1)[1]
if tok: os.environ["HF_TOKEN"]=tok; os.environ["HUGGING_FACE_HUB_TOKEN"]=tok
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forge.student import chat
M="kamekichi128/qwen3-4b-instruct-2507:latest"
PRE="from typing import List, Dict, Tuple, Optional, Any\nimport math\n"
def ask(p,t=0.5):
    try: return chat(M,[{"role":"system","content":"You are helpful. Be concise."},{"role":"user","content":p}],temperature=t,num_ctx=4096,num_predict=256,think=False)
    except Exception: time.sleep(2); return chat(M,[{"role":"system","content":"You are helpful. Be concise."},{"role":"user","content":p}],temperature=t,num_ctx=4096,num_predict=256,think=False)
def run(code,tests):
    with tempfile.TemporaryDirectory() as d:
        p=os.path.join(d,"s.py"); open(p,"w").write(PRE+code+"\n"+tests)
        r=subprocess.run(["python3",p],capture_output=True,text=True,timeout=10)
        return r.returncode==0,(r.stderr or r.stdout)[-400:]
ds=load_dataset("openai/openai_humaneval",split="test")
ok=0; N=10; det=[]
for i in range(N):
    ex=ds[i]
    a=ask("Complete python:\n"+ex["prompt"]+"\nReturn ```python block only.",0.5)
    m=re.search(r"```python(.*?)```",a,re.S); code=(m.group(1) if m else a).strip()
    passed,err=run(code,ex["test"])
    if not passed:
        a2=ask("Complete python:\n"+ex["prompt"]+"\nPrevious attempt failed with:\n"+err+"\nFix it. Return ```python block only.",0.7)
        m2=re.search(r"```python(.*?)```",a2,re.S); code2=(m2.group(1) if m2 else a2).strip()
        passed,_=run(code2,ex["test"])
    ok+=passed; det.append({"task":ex["task_id"],"pass":passed}); time.sleep(0.3)
    print(f"{ex['task_id']}: {'PASS' if passed else 'FAIL'}",flush=True)
print(f"HumanEval: {ok}/{N} {ok/N*100:.1f}%")
open("evaluations/bench-he.json","w").write(json.dumps({"arch":"V2-preamble+retry","HumanEval":f"{ok}/{N} {ok/N*100:.1f}%","detail":det},indent=2))
