# Tanglish Offensive-Language Classifier (LoRA Fine-Tune)

Fine-tuning a small open instruction-tuned LLM with QLoRA to classify
offensive language in Tamil-English code-mixed text ("Tanglish").

## Task and data
6-class offensive-language classification on the Tamil-English split of
DravidianCodeMix (Chakravarthi et al., 2021), using the dataset's
official train/dev/test split for direct comparability with published
baselines. Train: 35,139 | Dev: 4,388 | Test: 4,392 (before dedup).

## Week 1 findings
- Heavily imbalanced: Not_offensive ~72% of every split, rarest class
  (Offensive_Targeted_Insult_Other) ~1.3-1.6%. Distribution is
  consistent across all three splits, confirming it's real, not a
  sampling artifact.
- Training on the natural distribution; tracking macro F1 and
  per-class F1 alongside weighted F1, since weighted F1 can hide poor
  minority-class performance.
- Text mixes three patterns, sometimes in the same comment: Tamil in
  Latin script, Tamil in native script, English-Tamil word mixing.

## Roadmap
1. Week 1 - setup, official data, exploration (done)
2. Week 2 - TF-IDF+LogReg baseline, zero-shot LLM baseline
3. Week 3 - instruction-format the data
4. Week 4 - load base model in 4-bit, attach LoRA, dry run
5. Weeks 5-6 - full training runs, log with W&B
6. Week 7 - evaluation, error analysis
7. Week 8 - write-up, repo polish
