# FORGE-4B

Small-model agentic harness + distillation experiments on a 4B student (Qwen3-4B-Instruct-2507, Q4 via Ollama).
Status: **research snapshot, not a released model**. Base-model benchmarks are final; SFT is in progress after a
data-formatting bug invalidated the first adapter (details below — the old adapter is **not shipped**).

## What this is

- A dense ≤4.5B student + verifier/tools/PaCoRe harness for checkable work (coding, debugging, terminal, tool use).
- A 155k-row distillation corpus (harvested + synthetic) and QLoRA SFT pipeline for Kaggle T4×2.
- Honest small-n benchmarks with 95% CIs, harness details, and known limitations stated inline.

## Benchmarks as of 2026-09-19 (all local, Q4_K_M, Ollama, `FORGE_ZEN=0`)

> **Please do not read these as apple-to-apple comparisons with vendor numbers.** Our runs are 0-shot/single-letter
> (or few-shot where noted), Q4 quantized, n=20–42 subsets, temp 0.2 unless noted. Vendor numbers below are BF16,
> usually 5-shot/CoT, full test sets. Both are reported with their harnesses so readers can judge. CIs are wide —
> treat ±10–20pp as noise on n=20.

### Base 4B (`kamekichi128/qwen3-4b-instruct-2507:latest`), temp 0.2, single-shot

| Family | Score | n | 95% CI | Notes |
|---|---|---|---|---|
| MMLU-Pro first-40 | 11/40 27.5% | 40 | ±13.8pp | **All 40 are `business`** — MMLU-Pro test split is category-ordered; this is a business slice, not the benchmark |
| MMLU-Pro stratified (3×14 cats) | 16/42 **38.1%** | 42 | ±14.7pp | Category-balanced; flat per-category (no cat above 2/3). Use this number, not the 27.5% |
| GPQA-diamond (open mirror) | 5/20 25.0% | 20 | ±19.0pp | `fingertap/GPQA-Diamond`; official `Idavidrein/gpqa` is gated |
| GSM8K (strict `<a>` tags) | 8/30 26.7% | 30 | ±15.8pp | Format-strict; CoT/tool harness scores higher (see below) |
| HumanEval (strict exec + `typing` preamble) | 20/20 **100%** | 20 | ±0.0pp | First-20 subset; preamble is standard practice (real harnesses inject imports) |

### Same-technique check: Qwen-style few-shot, limit 20, temp 0.2 (`scripts/bench_apples_qwen.py`)

| Model | MMLU 5-shot | MMLU-Pro 5-shot | GSM8K 8-shot | HumanEval 0-shot |
|---|---|---|---|---|
| FORGE 4B Q4 | 9/20 45.0% | 13/20 **65.0%** | 4/20 20.0% | 20/20 100% |
| qwen3:1.7b Q4 (same script) | 8/20 40.0% | 5/20 25.0% | 6/20 30.0% | 20/20 100% |

GSM8K 20-vs-30 on n=20 is within noise (±20pp); the Pro gap (65 vs 25) is the signal. Few-shot closes most of the
gap to the vendor 5-shot figure below — the earlier 0-shot-vs-vendor gap was harness, not weights.

### Exploration harness (T 0.95, top_k 40, top_p 0.95 — for pass@k/voting, not pass@1)

| Model | MMLU-Pro strat-28 | GPQA 20 | GSM8K 20 | HumanEval 20 | HLE-20 lenient text-only |
|---|---|---|---|---|---|
| 4B | 13/28 46.4% ±18.5 | 6/20 30.0% | 4/20 20.0% | 20/20 → 20/20 | 1/20 5.0% |
| hari-ai 2.5B | 8/28 28.6% | 4/20 20.0% | 0/20 | 17/20 → 18/20 | 1/20 5.0% |
| qwen3:1.7b | 9/28 32.1% | 3/20 15.0% | 6/20 30.0% | 20/20 → 20/20 | 0/20 |

HLE is text-only here (images replaced with a placeholder note), n=20, lenient match — expect ~2–5% strict.
Frontier reference (different harness): DeepSeek-V4-Pro HLE 37.7. HLE needs tools+retrieval at 4B, not temperature.

### hari-ai 2.5B reference (`hari-ai:latest`, same 110 + strat)

MMLU-Pro 8/40 20.0% ±12.4 (business slice) / strat-42 10/42 23.8% ±12.9 · GPQA 1/20 5.0% · GSM8K 0/30 0.0%
(strict `<a>`-tag non-compliance, not proof of no math ability) · HumanEval 19/20 95.0% ±9.6.

### Arena-50 agentic proxy (older run, 2026-09-14)

pass@1 42/50 (84%), pass@5 48/50 (96%) vs cloud-9B single-shot 46/50 on the same 50. Pass@5 inflates via retry —
report pass@1 as the honest number.

### SFT status (honest)

- v3 (155k full) and v4 (40k cap) runs timed out on T4×2 throughput (~8–17 s/step).
- v5 resumed to epoch 1.0 (train loss 0.19) and saved a LoRA adapter — but its eval is **invalid**: the training
  template omitted `<|im_end|>`/EOS, so the model echoes transcript fragments. **That adapter is not shipped.**
- Fix applied in `colab/02_sft.py` (native `tokenizer.apply_chat_template` + EOS). Corrected 15k run
  (`forge-4b-fixed-sft` v2, maxlen 768, rank 16, 1 epoch) was in progress at time of writing.

## Vendor / published reference numbers (separate harness — not comparable to ours)

| Model | MMLU-Pro | GPQA | GSM8K | HumanEval | Source |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 (BF16) | 69.6 (5-shot) | 39.7 (agentic best-run) | 87.8 (think) | 67.0 (no-think) | Qwen3 report; CodeSOTA; PostTrainBench |
| Qwen3.6-35B-A3B family | 85.2 | 86.0 | ~92 | 73.4 (SWE proxy) | HF/BenchLM |
| Gemma-3 4B/12B/27B PT | 29.2/45.3/52.2 (CoT) | 15.0/25.4/24.3 | 38.4/71.0/82.6 | 36.0/45.7/48.8 | Google Gemma 3 report (PT table) |
| Gemma-3-4B-IT / 27B-IT | — / 76.9 (MMLU) | — | 89.2 / 95.9 | 71.3 / 87.8 | Google Gemma 3 report (IT) |
| Llama-3.2-3B-Inst / 3.1-8B-Inst | — (MMLU 63.4/69.4) | 32.8 / 32.8 | 77.7 / 84.5 | — | Meta model cards |
| Qwen2.5-14B/32B-Inst | 51.2 / 55.1 | — | 90.2 / 92.9 | 56.7 / 58.5 | Qwen2.5 blog |
| DeepSeek-R1-Distill-Qwen-1.5B | — | 33.8◆ | — (MATH-500 83.9) | — | R1 report arXiv:2501.12948 |
| MiniCPM-2.4B (base) | — (MMLU 53.5) | — | 53.8 | 50.0 | MiniCPM paper arXiv:2404.06395 |

◆ GPQA-diamond. PostTrainBench figures are best-of-many agentic runs, not single-shot.

## Reproduce (from `/home/dell/forge-4b`)

```bash
# plain chat (no harness)
ollama run kamekichi128/qwen3-4b-instruct-2507:latest
ollama run hari-ai:latest

# harness: single question (verifier + tools)
PYTHONPATH=/home/dell/forge-4b FORGE_ZEN=0 python3 -m forge.cli ask "write quicksort in python" --capability 04_coding
PYTHONPATH=/home/dell/forge-4b FORGE_ZEN=0 python3 -m forge.cli run --stage generate --problem "..." --model kamekichi128/qwen3-4b-instruct-2507:latest

# benches
PYTHONPATH=/home/dell/forge-4b FORGE_ZEN=0 python3 -m forge.bench.run --limit 50 --attempts 5 --track A
FORGE_ZEN=0 python3 scripts/real_bench_big.py
FORGE_ZEN=0 python3 scripts/bench_mmlu_strat.py
FORGE_ZEN=0 python3 scripts/bench_apples_qwen.py --model kamekichi128/qwen3-4b-instruct-2507:latest --limit 20
FORGE_ZEN=0 python3 scripts/bench_hot.py --model kamekichi128/qwen3-4b-instruct-2507:latest --limit 20 --temp 0.95 --top_p 0.95 --top_k 40
```

## Limitations

- Small-n subsets with wide CIs; MMLU-Pro first-n is a single-category slice (use stratified).
- Q4 quant costs a few pp vs BF16; temp-0.2 single-shot understates what few-shot/CoT/tools unlock.
- No SFT weights are published yet; the only trained adapter is invalid (template bug) and withheld.
- GPQA uses an open mirror (official set is gated); HLE here is text-only, n=20, lenient.

## Credits

- **Base models:** Qwen3 team (Alibaba) — Qwen3-4B-Instruct-2507 / Qwen3-1.7B (Apache-2.0); OpenBMB MiniCPM family
  (reference for the `hari-ai` 2.5B local model); `kamekichi128` Ollama quant of Qwen3-4B-Instruct-2507.
- **Published-number sources:** Qwen3 technical report (arXiv:2505.09388); Qwen2.5 blog; Google Gemma 3 report;
  Meta Llama 3.1/3.2 model cards; DeepSeek-R1 report (arXiv:2501.12948); MiniCPM paper (arXiv:2404.06395);
  Qwen2.5-Coder report (arXiv:2409.12186); CodeSOTA and PostTrainBench leaderboards; Artificial Analysis;
  llm-stats.com; HuggingFace model cards and community eval reports.
- **Datasets:** Jackrong `gpt-oss-120B-distilled-reasoning` and `Qwen3.5-reasoning-700x` (Apache-2.0);
  nohurry `Opus-4.6-Reasoning-3000x-filtered` (quarantined split); TIGER-Lab MMLU-Pro; `fingertap/GPQA-Diamond`
  mirror of the gated `Idavidrein/gpqa`; OpenAI GSM8K and HumanEval; CAIS HLE (`cais/hle`); `cais/mmlu`;
  `lighteval/MATH`; plus locally synthesized rows (`scripts/synth_gen.py`).
- **Tooling/infra:** Ollama; Unsloth + TRL + PEFT + HuggingFace Datasets; Kaggle T4×2 (dataset
  `harikrishnasriniv/forge-4b-src`, kernels); Python stdlib sandbox executor.
- **Methods:** PaCoRe parallel coordinated reasoning (ACL 2026 pattern); verifier-first agentic loop
  (SWE-smith / T1 / BOND style); MiniCPM small-model scaling practice.

## License

Code in this repo is shared as-is for research. Upstream model weights follow their own licenses
(Qwen3 Apache-2.0; Llama/Gemma community licenses; MiniCPM/OpenBMB terms). Benchmark datasets follow theirs
(GSM8K/HumanEval MIT; MMLU-Pro/GPQA/HLE per their cards). The quarantined split is excluded from any release.
