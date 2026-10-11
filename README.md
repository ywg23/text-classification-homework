# Text‑Classification‑Homework
Course assignment: A Comparative Study of Text Representations for News Classification

This repository contains source code for news classification experiments on the NYT dataset.
We compare bag‑of‑words, averaged static word embeddings and fine‑tuned BERT.

## Project Structure
```
text-classification-homework/
├── src/
│   ├── 01_prepare_data.py     # Dataset preprocessing & train/val/test split
│   ├── 02_bow.py              # Binary BoW / Word Frequency / TF‑IDF experiments
│   ├── 03_embeddings.py       # GloVe, Word2Vec(AG‑News), Word2Vec(NYT) experiments
│   ├── 04_bert.py             # BERT‑base‑uncased fine‑tuning
│   └── 05_merge_results.py    # Merge all experiment results
├── data/
│   └── processed/             # Saved split dataset (generated after running 01_prepare_data.py)
├── results/                   # Experiment output csv files (generated automatically)
├── models/                    # Saved Word2Vec & BERT model checkpoints
├── report/
│   ├── main.tex               # ACL‑style experiment report source
│   └── custom.bib             # BibTeX references
└── README.md
```

## Environment & Dependencies
Create python virtual environment (`.venv`):
```bash
python -m venv .venv

# activate venv
# Windows
.venv\Scripts\activate
# Linux / MacOS
source .venv/bin/activate

pip install scikit-learn==1.6.1 gensim==4.4.0 transformers==4.57.6 torch==2.8.0 nltk==3.9.2
```

### Package Versions
- scikit‑learn: `1.6.1`
- gensim: `4.4.0`
- transformers: `4.57.6`
- torch: `2.8.0`
- nltk: `3.9.2`

## Dataset
1. **NYT dataset**: Main classification dataset with `text` and `label` columns.
2. **AG‑News dataset**: Only used for training Word2Vec embeddings, not for classification training.

> Note: Raw dataset files are not included in this repo. Please download them and place under `data/` directory.

## Reproduce Experiments
All experiments use fixed random seed = 42.
NYT split ratio: train 80% / val 10% / test 10%.

Run scripts in sequence:
```bash
# Step1: data preprocessing
python src/01_prepare_data.py

# Step2: BoW experiments
python src/02_bow.py

# Step3: static embedding experiments (GloVe / Word2Vec)
python src/03_embeddings.py

# Step4: BERT fine‑tuning
python src/04_bert.py

# Step5: merge all test‑set results
python src/05_merge_results.py
```

### Word2Vec Hyper‑parameters
- vector_size=100
- window=5
- min_count=2
- negative=5
- sg=1 (Skip‑gram)
- epochs=5
- workers=4
- seed=42

> NYT‑trained Word2Vec is trained **only on NYT training split** to avoid data leakage.

### BERT Settings
- Model: `google‑bert/bert‑base‑uncased`
- max sequence length: 64
- training epochs: 3
- learning rate: $2\times10^{-5}$
- Hardware: CPU

## Evaluation Metrics
- Accuracy
- Macro‑F1 (for imbalanced multi‑class evaluation)

## Experiment Results
All outputs will be saved under `results/all_results.csv`.

| Method                                                   | Accuracy | Macro‑F1 |
| -------------------------------------------------------- | -------- | -------- |
| Binary BoW + Logistic Regression                         | 0.9843   | 0.9628   |
| Word Frequency + Logistic Regression                     | 0.9886   | 0.9725   |
| TF–IDF + Logistic Regression                             | 0.9860   | 0.9672   |
| GloVe 6B (100d) mean + Logistic Regression               | 0.9773   | 0.9498   |
| Word2Vec trained on AG News (100d) + Logistic Regression | 0.9729   | 0.9399   |
| Word2Vec trained on NYT (100d) + Logistic Regression     | 0.9747   | 0.9451   |
| BERT‑base‑uncased fine‑tuning                            | 0.9703   | 0.9354   |

## Generate PDF Report
Compile `report/main.tex` with pdfLaTeX + BibTeX (ACL style).

## Report Summary
We compare three families of text representations for news classification on imbalanced NYT dataset.
Sparse bag‑of‑words (Word Frequency) achieves best performance in our setup.
Simple averaging static embeddings and BERT fine‑tuning show relatively lower results.
Complex models do not guarantee better performance under fixed experimental constraints.
