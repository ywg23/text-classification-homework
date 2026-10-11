import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
NYT_FILE = ROOT / "data" / "nyt.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def load_nyt(path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}\n请把 NYT 数据文件放到 data/nyt.csv，"
            "并确保至少包含 text 和 label 两列。"
        )
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("目前支持 CSV、JSON、JSONL 格式；如文件格式不同，请修改读取函数。")

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(f"数据必须有 text 和 label 两列。当前列名：{list(df.columns)}")

    df = df[["text", "label"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if df["label"].nunique() < 2:
        raise ValueError("label 至少需要两个类别，才能进行分类。")
    return df

def main():
    df = load_nyt(NYT_FILE)
    # 先分出 10% 测试集，再从剩余 90% 中取 1/9 作为验证集，
    # 最终比例约为训练 80%、验证 10%、测试 10%。
    train_val, test = train_test_split(
        df, test_size=0.10, random_state=RANDOM_STATE, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val, test_size=(1/9), random_state=RANDOM_STATE,
        stratify=train_val["label"]
    )
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT_DIR / f"nyt_{name}.csv", index=False, encoding="utf-8-sig")
        print(f"{name}: {len(part)} 条")
    with open(OUT_DIR / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(sorted(df["label"].unique().tolist()), f, ensure_ascii=False, indent=2)
    print("\n类别数量：")
    print(df["label"].value_counts())
    print(f"\n数据划分已保存到：{OUT_DIR}")

if __name__ == "__main__":
    main()
