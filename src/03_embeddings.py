from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
DIM = 100
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")

def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())

def read_split(name):
    path = PROCESSED / f"nyt_{name}.csv"
    if not path.exists():
        raise FileNotFoundError("请先运行：python src/01_prepare_data.py")
    df = pd.read_csv(path).fillna("")
    return df["text"].astype(str).tolist(), df["label"].astype(str).tolist()

def mean_vectors(texts, keyed_vectors):
    X = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        vecs = [keyed_vectors[w] for w in tokenize(text) if w in keyed_vectors]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
        # 若一个文档没有词命中词表，保留全零向量
    return X

def load_glove(path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 GloVe 文件：{path}\n请下载并解压 glove.6B.zip，"
            "将 glove.6B.100d.txt 放到 data/glove/ 目录。"
        )
    vectors = {}
    print("正在读取 GloVe 文件，首次加载可能需要几分钟……")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != DIM + 1:
                continue
            try:
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
    if not vectors:
        raise ValueError("没有读到 GloVe 向量，请确认使用 glove.6B.100d.txt。")
    return vectors

def load_ag_texts(path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 AG News 数据：{path}\n请把 AG News 训练文本放到该路径。"
            "CSV 文件应包含 text 列；也支持只有一列文本的 CSV。"
        )
    df = pd.read_csv(path)
    if "text" in df.columns:
        texts = df["text"].dropna().astype(str).tolist()
    else:
        # 对没有表头的两列/多列 AG News CSV，通常正文可能在最后一列；
        # 请检查数据实际格式，必要时调整此处。
        df = pd.read_csv(path, header=None)
        if df.shape[1] >= 3:
            texts = (df.iloc[:, -2].fillna("").astype(str) + " " +
                     df.iloc[:, -1].fillna("").astype(str)).tolist()
        else:
            texts = df.iloc[:, -1].dropna().astype(str).tolist()
    return texts

def train_w2v(texts, model_path):
    sentences = [tokenize(t) for t in texts]
    sentences = [s for s in sentences if s]
    if not sentences:
        raise ValueError("Word2Vec 训练文本为空，请检查数据。")
    print(f"训练 Word2Vec，句子数量：{len(sentences)}")
    model = Word2Vec(
        sentences=sentences, vector_size=DIM, window=5,
        min_count=2, workers=4, sg=1, seed=42, epochs=5
    )
    model.save(str(model_path))
    return model.wv

def evaluate(name, train_texts, y_train, val_texts, y_val, test_texts, y_test, vectors):
    X_train = mean_vectors(train_texts, vectors)
    X_val = mean_vectors(val_texts, vectors)
    X_test = mean_vectors(test_texts, vectors)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    print(f"{name} 验证集 Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    mf1 = f1_score(y_test, pred, average="macro", zero_division=0)
    print(f"{name} 测试集 Accuracy: {acc:.4f}")
    print(f"{name} 测试集 Macro-F1: {mf1:.4f}")
    print(classification_report(y_test, pred, zero_division=0))
    joblib.dump(clf, MODELS / f"{name}_classifier.joblib")
    return {"method": name, "accuracy": acc, "macro_f1": mf1}

def main():
    x_train, y_train = read_split("train")
    x_val, y_val = read_split("val")
    x_test, y_test = read_split("test")
    rows = []

    glove_path = DATA / "glove" / "glove.6B.100d.txt"
    glove = load_glove(glove_path)
    rows.append(evaluate("glove_100d", x_train, y_train, x_val, y_val, x_test, y_test, glove))
    del glove

    ag_path = DATA / "ag.csv"
    ag_texts = load_ag_texts(ag_path)
    ag_vectors = train_w2v(ag_texts, MODELS / "word2vec_ag_news.model")
    rows.append(evaluate("word2vec_ag_news_100d", x_train, y_train, x_val, y_val, x_test, y_test, ag_vectors))

    # 为避免测试文本参与词向量训练，只使用 NYT 训练集训练 Word2Vec
    nyt_vectors = train_w2v(x_train, MODELS / "word2vec_nyt.model")
    rows.append(evaluate("word2vec_nyt_100d", x_train, y_train, x_val, y_val, x_test, y_test, nyt_vectors))

    out = RESULTS / "embedding_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存：{out}")

if __name__ == "__main__":
    main()
