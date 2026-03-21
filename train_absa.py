import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np


MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
ASPECTS = ["product", "price", "service", "variety"]
LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}
ID2LABEL = {0: "neg", 1: "neu", 2: "pos"}

MAX_LEN = 64
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


df = pd.read_csv("data/absa_mobile_shop.csv")

for a in ASPECTS:
    df[a] = df[a].astype(str).str.strip().str.lower().map(LABEL2ID)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ABSADataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        labels = torch.tensor([row[a] for a in ASPECTS], dtype=torch.long)  # สำคัญ
        item["labels"] = labels
        return item

train_loader = DataLoader(ABSADataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ABSADataset(val_df), batch_size=BATCH_SIZE, shuffle=False)


class MultiHeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList([nn.Linear(hidden, 3) for _ in ASPECTS])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # CLS token
        cls = self.dropout(cls)
        logits = torch.stack([head(cls) for head in self.heads], dim=1)  # (B,4,3)
        return logits

model = MultiHeadModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def eval_f1(model, loader):
    model.eval()
    all_true, all_pred = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)     # (B,4,3)
        pred = logits.argmax(dim=-1)                  # (B,4)

        all_true.append(labels.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    f1s = []
    for i in range(len(ASPECTS)):
        f1s.append(f1_score(y_true[:, i], y_pred[:, i], average="macro"))

    return float(np.mean(f1s))


best_f1 = -1.0

print("Device:", DEVICE)
print("Train size:", len(train_df), "Val size:", len(val_df))

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)

        loss = 0
        for i in range(len(ASPECTS)):
            loss += loss_fn(logits[:, i], labels[:, i])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_f1 = eval_f1(model, val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val_F1(mean): {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "absa_model.pt")
        print("✅ Saved BEST model -> absa_model.pt")

print("Training Finished ✅ | Best Val_F1(mean):", best_f1)