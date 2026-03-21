import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
ASPECTS = ["product", "price", "service", "variety"]
LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL = {0: "neg", 1: "neu", 2: "pos"}

MAX_LEN = 64       
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_PATH = "data/absa_mobile_shop.csv"
WEIGHTS_PATH = "absa_model.pt"

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ABSADataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = tokenizer(
            str(row["text"]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        labels = torch.tensor([row[a] for a in ASPECTS], dtype=torch.long)
        item["labels"] = labels
        return item



class MultiHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in ASPECTS])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        logits = torch.stack([head(cls) for head in self.heads], dim=1)  # (B,4,3)
        return logits


@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_true, all_pred = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        pred = logits.argmax(dim=-1)

        all_true.append(labels.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)  # (N,4)
    y_pred = np.concatenate(all_pred, axis=0)
    return y_true, y_pred


def save_confusion_matrix(cm, title, path):

    fig = plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0,1,2], ["neg","neu","pos"])
    plt.yticks([0,1,2], ["neg","neu","pos"])


    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Not found: {CSV_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Not found: {WEIGHTS_PATH} (ต้องเทรนก่อน)")

    df = pd.read_csv(CSV_PATH)

    for a in ASPECTS:
        df[a] = df[a].astype(str).str.strip().str.lower().map(LABEL2ID)

   
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True, stratify=df["service"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True, stratify=temp_df["service"])

    train_loader = DataLoader(ABSADataset(train_df), batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(ABSADataset(val_df),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(ABSADataset(test_df),  batch_size=BATCH_SIZE, shuffle=False)

 
    model = MultiHeadModel().to(DEVICE)
    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=True)


    y_true_val, y_pred_val = predict(model, val_loader)
    y_true_test, y_pred_test = predict(model, test_loader)

    def compute_metrics(y_true, y_pred):
        out = {}
        f1s = []
        accs = []
        for i, a in enumerate(ASPECTS):
            f1 = f1_score(y_true[:, i], y_pred[:, i], average="macro")
            acc = accuracy_score(y_true[:, i], y_pred[:, i])
            out[f"f1_macro_{a}"] = float(f1)
            out[f"acc_{a}"] = float(acc)
            f1s.append(f1)
            accs.append(acc)
        out["f1_macro_mean"] = float(np.mean(f1s))
        out["acc_mean"] = float(np.mean(accs))
        return out

    val_metrics = compute_metrics(y_true_val, y_pred_val)
    test_metrics = compute_metrics(y_true_test, y_pred_test)


    for i, a in enumerate(ASPECTS):
        cm = confusion_matrix(y_true_test[:, i], y_pred_test[:, i], labels=[0,1,2])
        save_confusion_matrix(
            cm,
            title=f"Confusion Matrix (TEST) - {a}",
            path=os.path.join(REPORT_DIR, f"cm_test_{a}.png")
        )

 
    lines = []
    lines.append("=== ABSA EVALUATION REPORT ===\n")
    lines.append(f"Device: {DEVICE}\n")
    lines.append(f"Data split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}\n")

    lines.append("\n--- VAL METRICS ---\n")
    for k, v in val_metrics.items():
        lines.append(f"{k}: {v:.4f}\n")

    lines.append("\n--- TEST METRICS ---\n")
    for k, v in test_metrics.items():
        lines.append(f"{k}: {v:.4f}\n")

    lines.append("\n=== CLASSIFICATION REPORT (TEST) PER ASPECT ===\n")
    for i, a in enumerate(ASPECTS):
        lines.append(f"\n[Aspect: {a}]\n")
        rep = classification_report(
            y_true_test[:, i], y_pred_test[:, i],
            target_names=[ID2LABEL[0], ID2LABEL[1], ID2LABEL[2]],
            digits=4
        )
        lines.append(rep + "\n")

    report_txt = os.path.join(REPORT_DIR, "report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

    summary = {
        "val": val_metrics,
        "test": test_metrics,
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "aspects": ASPECTS,
        "labels": ID2LABEL,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "model": MODEL_NAME,
        "weights": WEIGHTS_PATH
    }
    with open(os.path.join(REPORT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("✅ Saved reports to:", REPORT_DIR)
    print(" - reports/report.txt")
    print(" - reports/summary.json")
    print(" - reports/cm_test_*.png (4 files)")


if __name__ == "__main__":
    main()