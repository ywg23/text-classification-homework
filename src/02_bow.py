from pathlib import Path
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

def read_split(name):
    path = DATA / f"nyt_{name}.csv"
    if not path.exists():
        raise FileNotFoundError("请先运行：python src/01_prepare_data.py")
    df = pd.read_csv(path).fillna("")
    return df["text"].astype(str).tolist(), df["label"].astype(str).tolist()

def main():
    x_train, y_train = read_split("train")
    x_val, y_val = read_split("val")
    x_test, y_test = read_split("test")

    methods = {
        "binary_bow": CountVectorizer(binary=True, max_features=100000, ngram_range=(1, 1)),
        "word_frequency": CountVectorizer(binary=False, max_features=100000, ngram_range=(1, 1)),
        "tfidf": TfidfVectorizer(max_features=100000, ngram_range=(1, 2)),
    }
    rows = []
    for name, vectorizer in methods.items():
        print(f"\n========== {name} ==========")
        model = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs"))
        ])
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        print(f"验证集 Accuracy: {accuracy_score(y_val, val_pred):.4f}")
        pred = model.predict(x_test)
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        print(f"测试集 Accuracy: {acc:.4f}")
        print(f"测试集 Macro-F1: {macro_f1:.4f}")
        print(classification_report(y_test, pred, zero_division=0))
        joblib.dump(model, MODELS / f"{name}.joblib")
        rows.append({"method": name, "accuracy": acc, "macro_f1": macro_f1})
    out = RESULTS / "bow_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"结果已保存：{out}")

if __name__ == "__main__":
    main()
