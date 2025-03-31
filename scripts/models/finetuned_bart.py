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

model_path = os.path.join(current_dir, "../../pretrained_models/bart_results/checkpoint-1464")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")  
model = BartForSequenceClassification.from_pretrained(model_path)

test_encodings = tokenizer(test_df["text"].tolist(), truncation=True, padding=True, max_length=128)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
    def __len__(self):
        return len(self.encodings["input_ids"])

test_dataset = TestDataset(test_encodings)

trainer = Trainer(model=model)
predictions = trainer.predict(test_dataset)
logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
y_pred = logits.argmax(-1)
y_true = test_df["label"].tolist()

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
print(f"\nF1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

results_df = test_df.copy()
results_df["prediction"] = y_pred
results_df.to_csv("bart_finetuned_predictions.csv", index=False)
