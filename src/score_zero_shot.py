import pandas as pd
from sklearn.metrics import classification_report, f1_score

PREDICTIONS_FILE = "zero_shot_predictions.csv"

def main():
    df = pd.read_csv(PREDICTIONS_FILE)
    print(f"Total rows: {len(df)}")
    parse_fail_rate = (df["predicted_label"] == "PARSE_FAIL").mean()
    print(f"Parse failure rate: {parse_fail_rate:.1%}\n")
    weighted_f1 = f1_score(df["true_label"], df["predicted_label"], average="weighted")
    macro_f1 = f1_score(df["true_label"], df["predicted_label"], average="macro")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}\n")
    print(classification_report(df["true_label"], df["predicted_label"], zero_division=0))

if __name__ == "__main__":
    main()
