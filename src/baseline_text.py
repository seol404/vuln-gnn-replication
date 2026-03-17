import json
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def summarize(rows, name):
    labels = [r["label"] for r in rows]
    c = Counter(labels)
    print(f"\n== {name} ==")
    print(f"total: {len(rows)}")
    print(f"label counts: {c}  (0=good, 1=bad)")
    # quick peek at CWE distribution if name contains CWE...
    cwes = []
    for r in rows:
        n = r.get("name","")
        if "CWE" in n:
            # crude: take leading CWE token
            idx = n.find("CWE")
            cwes.append(n[idx:idx+6])  # e.g., CWE401
    if cwes:
        print("top CWE prefixes:", Counter(cwes).most_common(5))

def train_baseline(train_rows, test_rows):
    X_train = [r["code"] for r in train_rows]
    y_train = [r["label"] for r in train_rows]
    X_test  = [r["code"] for r in test_rows]
    y_test  = [r["label"] for r in test_rows]

    vec = TfidfVectorizer(
        lowercase=False,
        ngram_range=(1,2),
        max_features=50000
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    print("\n== Baseline TF-IDF + Logistic Regression ==")
    print("Test F1 (binary, label=1 as vulnerable):", f1_score(y_test, pred))
    print(classification_report(y_test, pred, digits=3))

def main():
    train_path = Path("outputs/train.jsonl")
    val_path   = Path("outputs/val.jsonl")
    test_path  = Path("outputs/test.jsonl")

    train_rows = load_jsonl(train_path)
    val_rows   = load_jsonl(val_path)
    test_rows  = load_jsonl(test_path)

    summarize(train_rows, "train")
    summarize(val_rows, "val")
    summarize(test_rows, "test")

    # train on train, evaluate on test
    train_baseline(train_rows, test_rows)

if __name__ == "__main__":
    main()