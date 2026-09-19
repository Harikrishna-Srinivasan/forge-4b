"""02_sft.py — Unsloth QLoRA SFT on variant-A (or --variant B) harvest.
Response-only training masked on assistant <think> (Jackrong recipe for Qwen3.5-2B,
ported to Qwen3-4B-Instruct-2507).
Run (A100; T4: lower --maxlen to 2048):
  python colab/02_sft.py --out models/forge-4b-code --expert code [--variant B]
Saves: LoRA adapter + merged 16-bit + (GGUF quant is a separate llama.cpp step).
"""
from __future__ import annotations
import argparse, glob, json, os

BASE = "unsloth/Qwen3-4B-Instruct-2507"


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_rows(patterns: list[str], variant: str, expert: str):
    rows = []
    want = "A-permissive" if variant == "A" else "B-quarantined" if variant == "B" else None  # AB = all combined
    for pat in patterns:
        # make repo-relative patterns work regardless of cwd (Kaggle: /kaggle/working/forge-4b)
        abs_pat = pat if os.path.isabs(pat) else os.path.join(ROOT, pat)
        for f in glob.glob(abs_pat):
            for line in open(f):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                v = d.get("variant", "A-permissive")
                if want is not None and v != want:
                    continue
                rows.append(d)
    return rows


def to_sharegpt(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        prob = r.get("problem") or r.get("prompt") or r.get("input") or ""
        ans = r.get("answer") or r.get("output") or r.get("completion") or ""
        if not prob or not ans:
            continue  # skip manifest/other non-training rows
        out.append({"conversations": [
            {"from": "human", "value": prob[:3000]},
            {"from": "gpt", "value": ans[:6000]}]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--expert", default="code")
    ap.add_argument("--variant", default="A", choices=["A", "B", "AB"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=0, help="cap rows (0=all) for T4 throughput")
    ap.add_argument("--resume", type=str, default="", help="checkpoint dir to resume from")
    a = ap.parse_args()

    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        BASE, max_seq_length=a.maxlen, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=a.rank, lora_alpha=a.rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth")
    from datasets import Dataset
    rows = load_rows(["datasets/harvest-*.jsonl", "datasets/synth-*.jsonl", "datasets/forge-*.jsonl"],
                     a.variant, a.expert)
    print(f"sft rows ({a.variant}): {len(rows)}")
    if a.limit and len(rows) > a.limit:
        # stratified by variant to keep harvest + synth mix
        import random
        random.seed(42)
        # keep all harvest (11k) then sample synth to fill limit
        harvest = [r for r in rows if r.get("teacher","").startswith("Jackrong") or "Opus" in r.get("teacher","")]
        synth = [r for r in rows if r not in harvest]
        random.shuffle(synth)
        need = a.limit - len(harvest)
        rows = harvest + synth[:max(0, need)]
        print(f"capped to {len(rows)} (harvest {len(harvest)} + synth {len(rows)-len(harvest)})")
    assert rows, "empty training set — run 01_harvest first"
    ds = Dataset.from_list(to_sharegpt(rows))

    def fmt(ex):
        h, g = ex["conversations"]
        # Use Qwen's native template. Hand-built ChatML previously omitted the
        # model's generation markers and taught it to echo transcript fragments.
        return {"text": tok.apply_chat_template(
            [{"role": "user", "content": h["value"]},
             {"role": "assistant", "content": g["value"]}],
            tokenize=False, add_generation_prompt=False)}
    ds = ds.map(fmt)
    from trl import SFTTrainer, SFTConfig
    # T4 throughput: per-device 4 + accum 4 (eff 16 single-GPU, 32 if DDP), 1024 len is ~2x faster
    SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, args=SFTConfig(
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        num_train_epochs=a.epochs, learning_rate=a.lr, max_seq_length=a.maxlen,
        output_dir=a.out, logging_steps=10, save_steps=500,
        save_total_limit=1, report_to="none", max_steps=-1,
        dataloader_num_workers=2, group_by_length=True)).train(
        resume_from_checkpoint=(a.resume or None))
    model.save_pretrained(a.out + "-lora")
    merged = model.merge_and_unload()
    merged.save_pretrained(a.out + "-merged")
    tok.save_pretrained(a.out + "-merged")
    print(f"saved {a.out}-lora + {a.out}-merged (variant {a.variant})")


if __name__ == "__main__":
    main()
