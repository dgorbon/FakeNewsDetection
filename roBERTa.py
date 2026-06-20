import pandas as pd
import numpy as np
import zipfile
import os

TRAIN_PATH = "train.tsv"
VALID_PATH = "valid.tsv"
TEST_PATH = "test.tsv"

TEXT_COL = "text"
LABEL_COL = "label"
TRUE_LIKE_LABELS = {"half-true", "mostly-true", "true"}

def load_liar_tsv(path: str) -> pd.DataFrame:
    columns = [
        "id",
        "original_label",
        "text",
        "subject",
        "speaker",
        "speaker_job",
        "state",
        "party",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",
    ]

    df = pd.read_csv(path, sep="\t", header=None, names=columns)

    normalized_label = df["original_label"].astype(str).str.strip().str.lower()
    df[LABEL_COL] = np.where(normalized_label.isin(TRUE_LIKE_LABELS), 0, 1).astype(int)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    return df[[TEXT_COL, LABEL_COL, "original_label"]]
    df = pd.read_csv(path, sep="\t", header=None, names=columns)

    normalized_label = df["original_label"].astype(str).str.strip().str.lower()
    df[LABEL_COL] = np.where(normalized_label.isin(TRUE_LIKE_LABELS), 0, 1).astype(int)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    return df[[TEXT_COL, LABEL_COL, "original_label"]]
train_df = load_liar_tsv(TRAIN_PATH)
valid_df = load_liar_tsv(VALID_PATH)
test_df = load_liar_tsv(TEST_PATH)

print("Train size:", len(train_df))
print("Valid size:", len(valid_df))
print("Test size :", len(test_df))
print("\nBinary label counts in train:")
print(train_df[LABEL_COL].value_counts().sort_index())
print("\nOriginal labels mapped to binary labels:")
print(train_df.groupby("original_label")[LABEL_COL].first().sort_index())

import os
import pandas as pd
import torch
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

# === Configuration ===
FILES = ["train.tsv", "test.tsv", "valid.tsv"]
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 16
MAX_LENGTH = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Model & Tokenizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = RobertaModel.from_pretrained(MODEL_NAME, output_hidden_states=True).to(device)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# === Load LIAR-style TSV ===
def load_tsv(file_path):
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=[ "id",
        "original_label",
        "text",
        "subject",
        "speaker",
        "speaker_job",
        "state",
        "party",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",]
    )

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    return df

# === Batch feature extraction ===
def extract_features_batch(texts):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # RoBERTa <s> token embedding, shape: [batch_size, 768]
    features = outputs.last_hidden_state[:, 0, :]

    return features.cpu()

# === Process one file ===
def run_file(file_name):
    if not os.path.exists(file_name):
        print(f"Skipping missing file: {file_name}")
        return

    split = file_name.replace(".tsv", "")
    out_path = os.path.join(OUTPUT_DIR, f"RoBERTa_{split}.npz")

    df = load_tsv(file_name)
    texts = df["text"].tolist()
    labels = df["label"].astype(str).tolist()
    ids = df["id"].astype(str).tolist()

    print(f"\n>>> Extracting {file_name}")
    print(f"Samples: {len(texts)}")

    all_features = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_features = extract_features_batch(batch_texts)
        all_features.append(batch_features)

    features = torch.cat(all_features, dim=0).numpy()

    np.savez_compressed(
        out_path,
        features=features,
        labels=np.array(labels),
        ids=np.array(ids),
        texts=np.array(texts)
    )

    print(f"Saved: {out_path}")
    print(f"Feature shape: {features.shape}")

# === Run all files ===
for file_name in FILES:
    run_file(file_name)

print("\nDone.")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

def load_model_tokenizer(model_path, use_fp16=True):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        output_hidden_states=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizer, device

def preprocess(text, tokenizer, device):
    return tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

def extract_features(text, model, tokenizer, device, pooling="mean"):
    model.eval()
    inputs = preprocess(text, tokenizer, device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden = outputs.hidden_states[-1]
    if pooling == "mean":
        mask = inputs["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        features = summed / summed_mask
    elif pooling == "cls":
        features = last_hidden[:, 0]
    else:
        raise ValueError("Unsupported pooling type: choose 'mean' or 'cls'")
    return F.normalize(features, p=2, dim=1)

def run_extraction(df_path, model_path, out_file, pooling="mean", text_col="Text"):
    df = pd.read_csv(df_path)
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)
    model, tokenizer, device = load_model_tokenizer(model_path)
    features = []
    for text in df[text_col]:
        emb = extract_features(text, model, tokenizer, device, pooling)
        features.append(emb)
    features_tensor = torch.cat(features, dim=0)
    np.savez_compressed(out_file, features_tensor.cpu().numpy())
    print(f"Saved: {out_file} with shape {features_tensor.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


train = np.load("outputs/RoBERTa_train.npz", allow_pickle=True)
valid = np.load("outputs/RoBERTa_valid.npz", allow_pickle=True)
test  = np.load("outputs/RoBERTa_test.npz", allow_pickle=True)

X_train = train["features"]
y_train = train["labels"]

X_valid = valid["features"]
y_valid = valid["labels"]

X_test = test["features"]
y_test = test["labels"]


clf = LogisticRegression(
    max_iter=5000,
    random_state=42
)

clf.fit(X_train, y_train)


valid_pred = clf.predict(X_valid)
print("Validation Accuracy:", accuracy_score(y_valid, valid_pred))


test_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

import os
import pandas as pd
import torch
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === Configuration ===
FILES = {
    "train": "train.tsv",
    "valid": "valid.tsv",
    "test": "test.tsv"
}

MODEL_NAME = "roberta-base"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 16
MAX_LENGTH = 512

TRUE_LIKE_LABELS = {"half-true", "mostly-true", "true"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load RoBERTa ===
model = RobertaModel.from_pretrained(MODEL_NAME).to(device)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# === Correct LIAR TSV Loader ===
def load_liar_tsv(path):
    columns = [
        "id",
        "original_label",
        "text",
        "subject",
        "speaker",
        "speaker_job",
        "state",
        "party",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",
    ]

    df = pd.read_csv(path, sep="\t", header=None, names=columns)

    df["text"] = df["text"].fillna("").astype(str)

    normalized = df["original_label"].astype(str).str.strip().str.lower()

    # 0 = true-ish, 1 = false-ish
    df["label"] = np.where(normalized.isin(TRUE_LIKE_LABELS), 0, 1).astype(int)

    return df[["id", "text", "label", "original_label"]]

# === Feature Extraction ===
def extract_features_batch(texts):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # RoBERTa <s> / CLS embedding
    features = outputs.last_hidden_state[:, 0, :]

    return features.cpu()

def extract_file(split, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = load_liar_tsv(file_path)

    texts = df["text"].tolist()
    labels = df["label"].to_numpy()
    ids = df["id"].astype(str).to_numpy()
    original_labels = df["original_label"].astype(str).to_numpy()

    print(f"\n>>> Extracting {split}")
    print(f"Samples: {len(texts)}")
    print("Label counts:")
    print(df["label"].value_counts().sort_index())

    all_features = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_features = extract_features_batch(batch_texts)
        all_features.append(batch_features)

    features = torch.cat(all_features, dim=0).numpy()

    out_path = os.path.join(OUTPUT_DIR, f"RoBERTa_{split}.npz")

    np.savez_compressed(
        out_path,
        features=features,
        labels=labels,
        ids=ids,
        original_labels=original_labels,
        texts=np.array(texts)
    )

    print(f"Saved: {out_path}")
    print(f"Feature shape: {features.shape}")

# === Extract embeddings ===
for split, path in FILES.items():
    extract_file(split, path)

# === Train classifier ===
train_data = np.load("outputs/RoBERTa_train.npz", allow_pickle=True)
valid_data = np.load("outputs/RoBERTa_valid.npz", allow_pickle=True)
test_data = np.load("outputs/RoBERTa_test.npz", allow_pickle=True)

X_train = train_data["features"]
y_train = train_data["labels"]

X_valid = valid_data["features"]
y_valid = valid_data["labels"]

X_test = test_data["features"]
y_test = test_data["labels"]

clf = LogisticRegression(
    max_iter=5000,
    random_state=42,
    class_weight="balanced"
)

print("\n>>> Training Logistic Regression classifier...")
clf.fit(X_train, y_train)

# === Evaluate ===
valid_pred = clf.predict(X_valid)
test_pred = clf.predict(X_test)

print("\nValidation Accuracy:", accuracy_score(y_valid, valid_pred))
print(classification_report(y_valid, valid_pred))

print("\nTest Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# === Save classifier ===
joblib.dump(clf, "outputs/roberta_logistic_regression_classifier.joblib")
print("\nSaved classifier to outputs/roberta_logistic_regression_classifier.joblib")
