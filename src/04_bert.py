from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models" / "bert_nyt"
RESULTS.mkdir(exist_ok=True)
MODELS.parent.mkdir(exist_ok=True)
MODEL_NAME = str(ROOT / "models" / "bert-base-uncased")
MAX_LENGTH = 64
EPOCHS = 3
SEED = 42

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LENGTH
        )
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

def read_split(name):
    path = DATA / f"nyt_{name}.csv"
    if not path.exists():
        raise FileNotFoundError("请先运行：python src/01_prepare_data.py")
    df = pd.read_csv(path).fillna("")
    return df["text"].astype(str).tolist(), df["label"].astype(str).tolist()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0)
    }

def main():
    set_seed(SEED)
    x_train, y_train = read_split("train")
    x_val, y_val = read_split("val")
    x_test, y_test = read_split("test")

    encoder = LabelEncoder()
    encoder.fit(y_train)
    unknown = set(y_val + y_test) - set(encoder.classes_)
    if unknown:
        raise ValueError(f"以下验证/测试类别未出现在训练集：{unknown}。请检查数据划分。")
    y_train_id = encoder.transform(y_train)
    y_val_id = encoder.transform(y_val)
    y_test_id = encoder.transform(y_test)

    # 获取分类类别总数，供 BERT 分类模型使用
    num_labels = len(encoder.classes_)

    print("分类类别数量：", num_labels)
    print("类别列表：", encoder.classes_)

    tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True
    )
    train_ds = NewsDataset(x_train, y_train_id, tokenizer)
    val_ds = NewsDataset(x_val, y_val_id, tokenizer)
    test_ds = NewsDataset(x_test, y_test_id, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    local_files_only=True
    )
    use_cuda = torch.cuda.is_available()
    print("使用设备：", "CUDA GPU" if use_cuda else "CPU")
    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "bert_checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        per_device_train_batch_size=8 if use_cuda else 4,
        per_device_eval_batch_size=16 if use_cuda else 4,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        fp16=use_cuda,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train()
    print("\n验证集指标：", trainer.evaluate(val_ds))
    pred_output = trainer.predict(test_ds)
    pred_ids = np.argmax(pred_output.predictions, axis=-1)
    acc = accuracy_score(y_test_id, pred_ids)
    mf1 = f1_score(y_test_id, pred_ids, average="macro", zero_division=0)
    print(f"\nBERT 测试集 Accuracy: {acc:.4f}")
    print(f"BERT 测试集 Macro-F1: {mf1:.4f}")
    print(classification_report(y_test_id, pred_ids, target_names=encoder.classes_, zero_division=0))

    trainer.save_model(str(MODELS))
    tokenizer.save_pretrained(str(MODELS))
    with open(MODELS / "label_classes.json", "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)
    pd.DataFrame([{
        "method": "bert_base_uncased",
        "accuracy": acc,
        "macro_f1": mf1
    }]).to_csv(RESULTS / "bert_results.csv", index=False, encoding="utf-8-sig")
    print(f"模型保存到：{MODELS}")
    print(f"结果保存到：{RESULTS / 'bert_results.csv'}")

if __name__ == "__main__":
    main()
