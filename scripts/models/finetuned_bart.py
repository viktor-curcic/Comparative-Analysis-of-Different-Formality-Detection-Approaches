import pandas as pd
import torch
import os
from transformers import (
    BartForSequenceClassification,
    BartTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
test_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/test.csv"))
train_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/train.csv"))
val_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/val.csv"))


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
model = BartForSequenceClassification.from_pretrained("facebook/bart-large", num_labels=2)

def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]}, labels

train_encodings, train_labels = tokenize_data(train_df["text"].tolist(), train_df["label"].tolist())
val_encodings, val_labels = tokenize_data(val_df["text"].tolist(), val_df["label"].tolist())
test_encodings, test_labels = tokenize_data(test_df["text"].tolist(), test_df["label"].tolist())

class FormalityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = FormalityDataset(train_encodings, train_labels)
val_dataset = FormalityDataset(val_encodings, val_labels)
test_dataset = FormalityDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir = None; #still creates a folder?? doesn't change anything, ignore
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir= None,
    seed=42
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

predictions = trainer.predict(test_dataset)
logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
y_pred = logits.argmax(-1)
y_true = test_df["label"].tolist()

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
print(f"\nF1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

results_df = test_df.copy()
results_df["prediction"] = y_pred
results_df.to_csv(os.path.join(current_dir, "../../results/bart_finetuned_predictions.csv"), index=False)
