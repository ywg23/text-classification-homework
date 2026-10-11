from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def main():
    files = [
        RESULTS / "bow_results.csv",
        RESULTS / "embedding_results.csv",
        RESULTS / "bert_results.csv",
    ]
    frames = []
    for path in files:
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"暂时未找到：{path}（对应实验尚未运行或未成功）")
    if not frames:
        raise FileNotFoundError("还没有任何结果文件。请先运行各项实验。")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RESULTS / "all_results.csv", index=False, encoding="utf-8-sig")
    print("\n========== 实验结果汇总 ==========")
    print(df.to_string(index=False))
    print(f"\n汇总表保存到：{RESULTS / 'all_results.csv'}")

if __name__ == "__main__":
    main()
