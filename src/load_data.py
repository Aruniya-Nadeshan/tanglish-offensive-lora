"""
Loads and cleans the official Tamil offensive-language splits from
DravidianCodeMix. Shared by the baseline script, the zero-shot eval,
and the LoRA fine-tuning script later.
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

LABELS = [
    "Not_offensive",
    "Offensive_Untargetede",
    "Offensive_Targeted_Insult_Individual",
    "Offensive_Targeted_Insult_Group",
    "Offensive_Targeted_Insult_Other",
    "not-Tamil",
]


def load_split(split):
    path = os.path.join(DATA_DIR, f"tamil_offensive_full_{split}.csv")
    df = pd.read_csv(
        path, sep="\t", header=None, names=["text", "label", "_extra"],
        on_bad_lines="skip",
    )
    df = df.drop(columns=["_extra"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"].isin(LABELS))]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def load_all():
    return load_split("train"), load_split("dev"), load_split("test")


if __name__ == "__main__":
    train, dev, test = load_all()
    print(f"Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")
