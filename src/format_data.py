"""
Week 3: formats the raw train/dev/test splits into instruction-style
JSONL files for SFTTrainer. Each line is a JSON object with two fields:
  - "text":  the full formatted conversation string (what the model trains on)
  - "label": the ground-truth label (kept for reference/debugging)

max_length decision: 95th percentile of token lengths is 351, 99th is 573.
Setting max_length=512 in training covers ~98% of examples without truncation.
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_all
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

LABEL_DESCRIPTIONS = """- Not_offensive: not offensive
- Offensive_Untargetede: offensive, but not aimed at a specific person/group
- Offensive_Targeted_Insult_Individual: offensive, aimed at a specific person
- Offensive_Targeted_Insult_Group: offensive, aimed at a specific group
- Offensive_Targeted_Insult_Other: offensive, aimed at something else (e.g. an event)
- not-Tamil: the comment is in a DIFFERENT language entirely (e.g. Hindi, French, Sinhala), \
with no Tamil content at all. Note: Tamil written purely in Tamil script, with no English \
mixed in, still counts as Tamil — NOT this category."""


def format_example(text, label, tokenizer):
    messages = [
        {
            "role": "user",
            "content": f"""Classify this Tamil-English code-mixed comment into exactly one category:
{LABEL_DESCRIPTIONS}

Comment: {text}

Respond with ONLY the category name."""
        },
        {
            "role": "assistant",
            "content": label
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def save_formatted(df, path, tokenizer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "text": format_example(row["text"], row["label"], tokenizer),
                "label": row["label"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(df)} examples to {path}")


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading data...")
    train, dev, test = load_all()

    save_formatted(train, "data/train_formatted.jsonl", tokenizer)
    save_formatted(dev,   "data/dev_formatted.jsonl",   tokenizer)
    save_formatted(test,  "data/test_formatted.jsonl",  tokenizer)
    print("Done.")


if __name__ == "__main__":
    main()
