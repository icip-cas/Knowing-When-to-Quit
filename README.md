# Knowing When to Quit: Diagnosing and Training LLMs to Abort Futile Reasoning

## Overview

When LLMs face tasks beyond their capability, they fail to recognize their own boundaries and persist in generating plausible-looking but fundamentally incorrect outputs — a phenomenon we term **futile reasoning**. This creates a critical reliability risk in high-stakes domains.

We introduce **CaRL** (**C**apability-**a**ligned **R**einforcement **L**earning), which aligns model behavior with its true capability boundary via two mechanisms:

1. **Capability-Calibrated Reward Shaping** — a strict reward hierarchy that incentivizes refusal over futile reasoning: *correct* (+1) > *valid refusal* (+0.5) > *hallucination* (-1).
2. **Hindsight Refusal Augmentation** — converts futile reasoning traces into refusal-format training samples to enrich the refusal signal.

CaRL-8B: https://huggingface.co/xinyan233333/CaRL-8B 

CaRL-14B: https://huggingface.co/xinyan233333/CaRL-14B 

VERL framework is based on commit  c0a740c7

---

## Repository Structure

```
CaRL/
├── data/
│   ├── make_data.py             # Generate Countdown datasets (N=3–8)
│   ├── make_sudoku.py           # Generate Sudoku eval set
│   ├── unk_train_{N}number.parquet   # Training splits
│   └── unk_eval_{N}number.jsonl     # Evaluation splits
├── scripts/
│   ├── train/
│   │   ├── qwen3_8b.sh          # Training script for Qwen3-8B
│   │   └── qwen3_14b.sh         # Training script for Qwen3-14B
│   └── eval/
│       ├── sample.py            # Sample model responses via vLLM/SGLang server
│       └── cal_reward.py        # Compute evaluation metrics
└── VERL-1222/                   # VERL training framework
```

---

## Tasks

**Countdown (N=3–8):** Given N numbers and a target, write an arithmetic expression using all N numbers (each exactly once) that equals the target. Difficulty scales with N.

**Sudoku:** Standard 9×9 puzzles with 30–50 empty cells, used as an out-of-distribution generalization test.

---

## Data Generation

```bash
python data/make_data.py      # Countdown: 1000 train + 100 eval per level (N=3–8)
python data/make_sudoku.py    # Sudoku: 100 eval samples
```

---

## Training

```bash
bash scripts/train/qwen3_8b.sh    # Set VERL_PATH and MODEL_PATH in the script first
bash scripts/train/qwen3_14b.sh
```

Key hyperparameters: GRPO, SGLang rollout, n=16, max_response_len=10240, lr=3e-6, kl_coef=0.001, 20 epochs, 8×GPU (tp=4).

---

## Evaluation

### 1. Sample Model Responses

```bash
python scripts/eval/sample.py \
  --input_file data/unk_eval_8number.jsonl \
  --output_dir ./output \
  --remote_url http://server1:port http://server2:port \
  --num_processes 64 \
  --n 16 \
  --temperature 0.6 \
  --generate_max_length 32768 \
  --tokenizer_name path/to/tokenizer
```

The script supports parallel processing, automatic resume, and multiple server endpoints for load balancing.

### 2. Compute Metrics

```bash
python scripts/eval/cal_reward.py
```

| Metric | Description |
|---|---|
| **Acc** | Fraction of problems answered correctly |
| **Rel Score** | Reliability score = (correct + 0.5 × valid_refusals) / total |
| **Ref Rate** | Fraction of problems where the model issued a refusal |
| **Halluc Rate** | Among wrong answers, fraction that were not refused (hallucinated) |
| **Len** | Average response length |
