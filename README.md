# Tanglish Offensive Language Classifier

Fine-tuning `Qwen2.5-3B-Instruct` with QLoRA to classify offensive language
in Tamil-English code-mixed text ("Tanglish") — the way people actually write
on social media in Tamil-speaking communities.

**Live demo:** [coming soon — Vercel deployment]
**Trained adapter:** [Aru-Niya/tanglish-offensive-lora](https://huggingface.co/Aru-Niya/tanglish-offensive-lora)

---

## Task

6-class offensive language classification on the Tamil-English split of
[DravidianCodeMix](https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset)
(Chakravarthi et al., 2021) — 44,000 manually annotated YouTube comments.

| Class | % of train |
|-------|-----------|
| Not_offensive | 72.3% |
| Offensive_Untargetede | 8.3% |
| Offensive_Targeted_Insult_Group | 7.3% |
| Offensive_Targeted_Insult_Individual | 6.7% |
| not-Tamil | 4.1% |
| Offensive_Targeted_Insult_Other | 1.3% |

Official train/dev/test split used throughout — results are directly
comparable to published baselines.

---

## Why this is hard

Tanglish mixes three patterns, often in the same comment:
- Tamil written in Latin script: `Enna da ellam avan seyal Mari iruku`
- Tamil written in native script: `தென்காசி மாவட்டம் நாடார் சமுதாயம்`
- Genuine English-Tamil word mixing: `Vera level BGM .. semma trailer`

Distinguishing offensive subtypes (targeted at individual vs. group vs.
untargeted) requires understanding social and linguistic context that
general pre-training does not encode.

---

## Results

| Model | Weighted F1 | Macro F1 |
|-------|-------------|----------|
| TF-IDF + Logistic Regression (baseline) | 0.676 | 0.426 |
| Zero-shot Qwen2.5-3B-Instruct | 0.569 | 0.161 |
| **QLoRA fine-tuned Qwen2.5-3B-Instruct** | **0.648** | **0.252** |

Fine-tuning improved over zero-shot by **56% on macro F1** (0.161 → 0.252).
The model did not surpass TF-IDF on macro F1 — analysis below explains why.

### Per-class breakdown (fine-tuned model, 500-example stratified test sample)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Not_offensive | 0.75 | 0.98 | 0.85 |
| not-Tamil | 1.00 | 0.33 | 0.50 |
| Offensive_Targeted_Insult_Individual | 0.29 | 0.11 | 0.16 |
| Offensive_Targeted_Insult_Group | 0.00 | 0.00 | 0.00 |
| Offensive_Untargetede | 0.00 | 0.00 | 0.00 |
| Offensive_Targeted_Insult_Other | 0.00 | 0.00 | 0.00 |

### Key finding

The model learned the dominant class (`Not_offensive`, recall 0.98) at the
expense of rare offensive subtypes. Three of six classes scored F1 = 0.00 —
the same pattern seen in the zero-shot baseline, though less severe. The
bottleneck is the training data distribution (72% Not_offensive), not the
model architecture. The natural next experiment is class-weighted loss,
which penalizes minority-class mistakes more heavily during training.

This finding — that fine-tuning improves zero-shot performance but class
imbalance prevents learning rare offensive categories without targeted
reweighting — is the primary result of this project.

---

## Approach

### Data cleaning (`src/load_data.py`)
- Dropped trailing empty column present in raw CSV files
- Stripped whitespace from labels
- Removed exact duplicate comments
- Used official train/dev/test split for direct comparability with literature

### Baseline 1: TF-IDF + Logistic Regression (`src/baseline_tfidf.py`)
Default `TfidfVectorizer` + `LogisticRegression(class_weight='balanced')`.
Vocabulary: 47,693 tokens — roughly 2× a monolingual English dataset of
similar size, explained by dual-script Tamil plus transliteration variants.
Unlike the original paper's baseline, we kept native Tamil script rather
than stripping non-Latin characters.

### Baseline 2: Zero-shot prompting (`src/zero_shot_baseline.py`)
`Qwen2.5-3B-Instruct` with a few-shot prompt (3 training examples).
Two bugs found and fixed during this step:
1. Native-script Tamil was systematically mislabeled as `not-Tamil` — fixed
   with a few-shot example demonstrating the distinction.
2. Model output `Not-offensive` (hyphen) while our label uses underscore —
   fixed by normalizing both sides before string matching.

### Fine-tuning (`src/train_lora.py`)
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Method: QLoRA (4-bit quantization + LoRA adapters)
- LoRA config: r=16, alpha=32, target modules: q/k/v/o projections
- Trainable parameters: 7.37M / 3.09B total (0.24%)
- Training data: 30% stratified sample of train split (10,466 examples)
- Steps: 300 (resource-constrained — free-tier GPU)
- Effective batch size: 16 (2 per device × 8 gradient accumulation steps)
- Learning rate: 2e-4 with cosine decay
- Experiment tracking: Weights & Biases

---

## Repository structure
src/

load_data.py          shared data loading and cleaning

baseline_tfidf.py     TF-IDF + Logistic Regression baseline

zero_shot_baseline.py zero-shot LLM evaluation

score_zero_shot.py    scoring script for zero-shot predictions

format_data.py        instruction-format data for SFTTrainer

train_lora.py         QLoRA fine-tuning script

data/

tamil_offensive_full_train.csv

tamil_offensive_full_dev.csv

tamil_offensive_full_test.csv

*_formatted.jsonl     instruction-formatted versions

zero_shot_predictions.csv

finetuned_predictions.csv
---

## Future work

- Class-weighted loss to improve minority-class performance
- Full training run (655 steps, full training set, 3 epochs)
- Comparison against Tamil-LLaMA (Tamil-specialized base model)
- Offensive-language task from the same dataset (separate label set)
- Submission to DravidianLangTech workshop (ACL/COLING)

---

## Citation

```bibtex
@inproceedings{chakravarthi2021findings,
  title={Findings of the Shared Task on Offensive Language Identification
         in Tamil, Malayalam, and Kannada},
  author={Chakravarthi, Bharathi Raja and others},
  booktitle={Proceedings of the First Workshop on Speech and Language
             Technologies for Dravidian Languages},
  year={2021}
}
```
