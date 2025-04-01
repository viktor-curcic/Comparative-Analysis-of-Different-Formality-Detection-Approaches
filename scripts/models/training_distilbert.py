import pandas as pd
import torch
import os
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
test_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/test.csv"))
train_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/train.csv"))
val_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/val.csv"))

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def tokenize_data(texts, labels=None):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return encodings, labels

train_encodings, train_labels = tokenize_data(train_df["text"].tolist(), train_df["label"].tolist())
val_encodings, val_labels = tokenize_data(val_df["text"].tolist(), val_df["label"].tolist())
test_encodings, test_labels = tokenize_data(test_df["text"].tolist(), test_df["label"].tolist())

class FormalityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = FormalityDataset(train_encodings, train_labels)
val_dataset = FormalityDataset(val_encodings, val_labels)
test_dataset = FormalityDataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir=None,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    seed=42,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

predictions = trainer.predict(test_dataset)
y_pred = predictions.predictions.argmax(-1)
y_true = test_df["label"].tolist()

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
print(f"\nF1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

results_df = test_df.copy()
results_df["prediction"] = y_pred
results_df.to_csv(os.path.join(current_dir, "../../results/distilbert_predictions.csv"), index=False)
