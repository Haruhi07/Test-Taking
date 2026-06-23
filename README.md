# Test-Taking

**Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies**

[![ACL 2026](https://img.shields.io/badge/ACL-2026-4b44ce?style=flat-square)]()
[![arXiv](https://img.shields.io/badge/arXiv-2605.29712-b31b1b?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)]()

This repository contains the official implementation of the paper *"Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies"*, accepted at **ACL 2026**.

> **Abstract:** Grounded claim factuality checking is important for LLM applications such as retrieval-augmented generation, as it helps users assess the correctness of generated outputs. We reframe grounded claim factuality checking as a true/false reading comprehension task and prompt LLMs with explicit test-taking strategies for efficient reasoning. Our method reduces token usage by over 80% compared to unguided open-ended reasoning, while achieving competitive performance across two factuality benchmarks — setting a new state of the art on one. To further reduce inference cost, we train small language models (SLMs) via SFT + DPO that perform on par with strong baselines while generating supporting rationales for interpretability.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Evaluation](#evaluation)
  - [LLM-based Evaluation](#llm-based-evaluation)
  - [SLM-based Evaluation](#slm-based-evaluation)
  - [SLM Claim Decomposition Evaluation](#slm-claim-decomposition-evaluation)
- [Training](#training)
  - [Claim Decomposition (SFT)](#claim-decomposition-sft)
  - [Fact Checking Format Alignment (SFT)](#fact-checking-format-alignment-sft)
  - [Mistake Revision (DPO)](#mistake-revision-dpo)
- [Results](#results)
- [Citation](#citation)

---

## Overview

Our work introduces a human-inspired test-taking strategy for grounded claim factuality checking. The key ideas are:

- **Claim Decomposition**: Break each claim into independent atomic facts, simplifying downstream verification.
- **Structured Fact Checking**: Apply a step-by-step criterion-based reasoning strategy — similar to how human examinees handle True/False reading comprehension questions — to verify each atomic fact against the grounding document.
- **SLM Distillation**: Train a small language model (Qwen3-0.6B) to replace the LLM in the evaluation pipeline, using a two-stage procedure: (1) SFT for format alignment, and (2) DPO for mistake revision.

The result is a factuality checking pipeline that is both **efficient** (over 80% fewer tokens than unguided CoT) and **interpretable** (produces rationales alongside judgements), with the SLM variant achieving performance comparable to much larger models.

---

## Pipeline

The evaluation consists of two steps:

```
                        ┌──────────────────────────────────────┐
                        │           Input Claim                │
                        └──────────────┬───────────────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │   Step 1: Claim Decomposer  │
                        │   (LLM or trained SLM)      │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                           ┌─────────────────────┐
                           │  Atomic Fact 1      │
                           │  Atomic Fact 2      │
                           │  Atomic Fact N      │
                           └───────┬─────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Step 2: Fact Checker per Atomic Fact                            │
│                                                                  │
│  C1 — Are the subject and object mentioned in the document?      │
│  C2 — Are their descriptions verifiable?                         │
│  C3 — Is their relationship explicitly stated?                   │
│  C4 — Can unverified information be inferred?                    │
│                                                                  │
│  Output: "Final Answer: yes" / "Final Answer: no"                │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
               ┌───────────────────────────────────┐
               │  Aggregation: Claim is True iff   │
               │  ALL atomic facts are True        │
               └───────────────────────────────────┘
```

The criteria C1–C3 guide the model to locate explicit evidence in the grounding document, while C4 enables reasonable inference. Applied sequentially, they form a structured decision process that mirrors how human test-takers work through True/False reading exercises.

---

## Repository Structure

```
Test-Taking/
├── main.py                   # LLM-based evaluation entry point
├── reason-eval.py            # Fact checking evaluation (BAcc, confusion matrix)
├── decompose-eval.py         # Decomposition quality evaluation (ROUGE-L, SECS)
├── utils.py                  # Shared prompts and utility functions
├── rebuttal.py               # Error analysis / rebuttal experiments
├── run-reason-eval.sh        # Slurm script for reason-eval
├── run-decompose-eval.sh     # Slurm script for decompose-eval
│
├── decompose-train/          # Claim decomposition SLM training
│   ├── train.py              # SFT training script
│   ├── make_dataset.py       # Generate synthetic data using teacher LLM
│   ├── combine_dataset.py    # Combine multi-source datasets
│   ├── run-train.sh
│   ├── run-make_dataset.sh
│   ├── run-combine_dataset.sh
│   └── generated_facts/      # Pre-generated decomposition data
│
├── SFT0.6B/                  # Fact checking SFT training (Qwen3-0.6B)
│   ├── train.py              # SFT for criterion-based reasoning
│   ├── combine_dataset.py    # Merge FacTax + LLM-AggreFact data
│   └── run-train.sh
│
├── DPO/                      # Fact checking DPO training
│   ├── train.py              # DPO for mistake revision
│   ├── make_dpo_dataset.py   # Prepare preference pairs
│   └── run-train.sh
│
└── README.md
```

---

## Requirements

- Python 3.10+
- [vLLM](https://github.com/vllm-project/vllm) — LLM inference
- [Transformers](https://github.com/huggingface/transformers)
- [TRL](https://github.com/huggingface/trl) — SFT and DPO training
- [Datasets](https://github.com/huggingface/datasets)
- [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers) — SECS evaluation
- [Evaluate](https://github.com/huggingface/evaluate) — ROUGE-L
- [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/)

Install dependencies:

```bash
pip install vllm transformers trl datasets sentence-transformers evaluate pandas scikit-learn
```

---

## Evaluation

### LLM-based Evaluation

Run the full two-stage pipeline (claim decomposition + fact checking):

```bash
python main.py \
    --model_path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --dataset_path lytang/LLM-AggreFact \
    --subset CNNDM \
    --output_path ./results
```

**Arguments:**

| Argument | Description |
|---|---|
| `--model_path` | HuggingFace model path or local checkpoint |
| `--dataset_path` | Dataset path (HuggingFace or local CSV) |
| `--batch_size` | Batch size (default: 16) |
| `--output_path` | Directory to save prediction CSV files |
| `--subset` | Subset name (e.g. CNNDM, XSUM) |
| `--origin` | Optional origin filter for AggreFact data |

**Supported datasets:**

- **LLM-AggreFact**: `lytang/LLM-AggreFact` (use `--subset` to specify subset)
- **FacTax-Benchmark**: Provide a local CSV path via `--dataset_path`
- **DiaSummFact**: Provide `diasumfact.csv` via `--dataset_path`

The script saves a CSV with a `preds` column (0 = False, 1 = True).

### SLM-based Evaluation

After training, run the same `main.py` with your trained checkpoint:

```bash
python main.py \
    --model_path /path/to/sft-checkpoint \
    --dataset_path lytang/LLM-AggreFact \
    --subset CNNDM \
    --output_path ./results/slm
```

Then score the results:

```bash
python reason-eval.py --folder_path ./results/slm
```

### SLM Claim Decomposition Evaluation

Evaluate decomposition quality with ROUGE-L and SECS:

```bash
python decompose-eval.py \
    --reference_path /path/to/reference/facts \
    --prediction_path /path/to/predicted/facts
```

Each directory should contain CSV files with a `facts` column.

---

## Training

We train three separate SLMs:
1. **Claim Decomposition** (SFT)
2. **Fact Checking – Reason Format Alignment** (SFT)
3. **Fact Checking – Mistake Revision** (DPO)

All training uses **Qwen3-0.6B** as the base model and **Qwen3-30B-A3B-Instruct** as the teacher.

### Claim Decomposition (SFT)

**Step 1: Generate teacher decomposition data**

```bash
cd decompose-train
python make_dataset.py
```

This script uses the teacher LLM to generate atomic facts from sentences sampled from grounding documents and existing claims, creating the training dataset.

**Step 2: Train the decomposition SLM**

```bash
python train.py \
    --dataset_path ./generated_facts/dataset \
    --output_path ./decompose-model
```

### Fact Checking Format Alignment (SFT)

**Step 1: Generate reasoning data**

The SFT training data is pre-generated by prompting the teacher LLM with altered facts so the model learns to apply the test-taking criteria.

**Step 2: Train the fact checking SLM (format alignment)**

```bash
cd SFT0.6B
python train.py \
    --dataset_path /path/to/reason-sft-data \
    --model_name Qwen/Qwen3-0.6B \
    --output_path ./reason-sft-model
```

### Mistake Revision (DPO)

**Step 1: Prepare preference pairs**

```bash
cd DPO
python make_dpo_dataset.py
```

This filters cases where the SFT student model produces incorrect judgements while the teacher produces correct ones, creating a preference dataset.

**Step 2: Run DPO training**

```bash
python train.py \
    --model_name /path/to/sft-checkpoint \
    --dataset_path /path/to/dpo-dataset.csv \
    --output_path ./dpo-model
```

**Key hyperparameters:**

| Stage | Learning Rate | Epochs | Gradient Accumulation |
|---|---|---|---|
| Decomposition SFT | 5e-6 | 10 | 4 |
| Reason SFT | 1e-5 | 5 | 16 |
| DPO | 1e-7 | 3 | 64 |

---

## Results

### LLM-based pipeline on FacTax-Benchmark

| Model | Size | Avg BAcc | Avg Rank |
|---|---|---|---|
| FACTAX-GPT-4o | — | 75.3 | 5.4 |
| **Qwen3-30B-A3B-Instruct (Ours)** | **30B** | **78.0** | **3.6** |
| Qwen3-4B-Instruct (Ours) | 4B | 73.0 | 7.1 |

Our 30B pipeline achieves **state-of-the-art** on FacTax-Benchmark.

### LLM-based pipeline on LLM-AggreFact

| Model | Size | Avg BAcc | Avg Rank |
|---|---|---|---|
| MiniCheck-BeSpoke-7B | 7B | 77.4 | 3.3 |
| **Qwen3-30B-A3B-Instruct (Ours)** | **30B** | **76.3** | **4.0** |
| GPT-4o-2024-05-13 | — | 75.9 | 4.8 |

Our method ranks **second** while being a zero-shot approach (no training data required).

### SLM pipeline (Qwen3-0.6B)

| Method | FacTax-Benchmark | LLM-AggreFact |
|---|---|---|
| Base (untrained) | 56.6 | 56.3 |
| + SFT | 64.8 | 71.3 |
| **+ SFT + DPO** | **72.6** | **73.6** |

The trained SLM (0.6B parameters) outperforms ChatGPT-3.5-based baselines and achieves performance comparable to TrueTeacher (11B parameters, 540B teacher).

### Token Efficiency

| Model | Mode | Avg Tokens Used | vs. Thinking |
|---|---|---|---|
| Qwen3-4B | Instruct | 2,803 | 10.4% |
| Qwen3-4B | Thinking | 27,001 | — |
| Qwen3-30B | Instruct | 1,152 | 10.5% |
| Qwen3-30B | Thinking | 10,924 | — |

Our structured prompt reduces token usage by **~90%** compared to unguided thinking mode on FacTax-Benchmark, and over **80%** on LLM-AggreFact.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{ye2026teaching,
  title={Teaching Language Models to Check Grounded Claim Factuality with Human Test-Taking Strategies},
  author={Ye, Yuxuan and Santos-Rodriguez, Raul and Simpson, Edwin},
  journal={arXiv preprint arXiv:2605.29712},
  year={2026}
}
```

---

## Acknowledgments

This work was supported by the China Scholarship Council (No. 202108060154), the Isambard-AI National AI Research Resource (AIRR), and the UKRI Turing AI Fellowship EP/V024817/1.
