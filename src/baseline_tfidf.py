import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_all
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

def main():
    train, dev, test = load_all()
    print(f"Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}\n")

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train["text"])
    X_dev = vectorizer.transform(dev["text"])
    X_test = vectorizer.transform(test["text"])

    clf = LogisticRegression(solver="newton-cg", class_weight="balanced", max_iter=1000)
    clf.fit(X_train, train["label"])

    for name, X, y in [("DEV", X_dev, dev["label"]), ("TEST", X_test, test["label"])]:
        preds = clf.predict(X)
        print(f"=== {name} ===")
        print(f"Weighted F1: {f1_score(y, preds, average='weighted'):.4f}")
        print(f"Macro F1:    {f1_score(y, preds, average='macro'):.4f}\n")
        print(classification_report(y, preds, zero_division=0))

if __name__ == "__main__":
    main()
