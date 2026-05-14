import os
os.environ["HF_HOME"] = "/home/pdelgado010/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/pdelgado010/.cache/huggingface"

import numpy as np
import pandas as pd
import torch
import sys
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset as TorchDataset


# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def load_data(fnames):
    data = []
    for fname in fnames:
        data.append(pd.read_csv(fname, sep=',', encoding='utf-8'))
    data = pd.concat(data)
    return data

# Label mapping
label2id = {"FAVOR": 0, "AGAINST": 1}
id2label = {0: "FAVOR", 1: "AGAINST"}

def encode_labels(df):
    df = df.copy()
    df["label"] = df["Stance"].map(label2id)
    return df

class StanceDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

def tokenize(df, tokenizer):
    texts = (df["Tweet"] + " [SEP] " + df["Target"]).tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return encodings

def evaluate_with_trainer(trainer, test_df, tokenizer, target_name):
    print(f"=== Fine-tuned | {target_name} ===")
    test_df = encode_labels(test_df)
    test_encodings = tokenize(test_df, tokenizer)
    test_dataset = StanceDataset(test_encodings, test_df["label"].tolist())
    
    # El trainer predice todo el dataset en batches automáticamente a velocidad máxima
    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    
    ground_truth = [id2label[l] for l in test_df["label"].tolist()]
    predictions = [id2label[p] for p in preds]
    
    print(classification_report(ground_truth, predictions, zero_division=0))

# ── DATA ──────────────────────────────────────────────────────────────────────

train_file_trump  = "PStance data/raw_train_trump.csv"
val_file_trump    = "PStance data/raw_val_trump.csv"


training_trump = encode_labels(load_data([train_file_trump]))
val_trump      = encode_labels(load_data([val_file_trump]))


test_file_trump  = "PStance data/raw_test_trump.csv"
test_file_biden  = "PStance data/raw_test_biden.csv"
test_file_bernie = "PStance data/raw_test_bernie.csv"

testing_trump  = load_data([test_file_trump])
testing_biden  = load_data([test_file_biden])
testing_bernie = load_data([test_file_bernie])

# ── TOKENIZER ─────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_encodings = tokenize(training_trump, tokenizer)
val_encodings   = tokenize(val_trump, tokenizer)

train_dataset = StanceDataset(train_encodings, training_trump["label"].tolist())
val_dataset   = StanceDataset(val_encodings, val_trump["label"].tolist())

# ── MODEL ─────────────────────────────────────────────────────────────────────

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-large",
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)

# ── TRAINING ──────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir="./roberta-stance",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=True,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,

    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
trainer.save_model("./roberta-stance-trump")
tokenizer.save_pretrained("./roberta-stance-trump")

# ── EVALUATION ────────────────────────────────────────────────────────────────

evaluate_with_trainer(trainer, testing_trump, tokenizer, "Trump (in-target)")
evaluate_with_trainer(trainer, testing_biden, tokenizer, "Biden (cross-target)")
evaluate_with_trainer(trainer, testing_bernie, tokenizer, "Bernie (cross-target)")