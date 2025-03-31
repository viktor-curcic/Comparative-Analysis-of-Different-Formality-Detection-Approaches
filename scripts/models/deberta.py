import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
import matplotlib.pyplot as plt

model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, max_length=512)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt",
    hypothesis_template="This text is {}.",
)

candidate_labels = ["informal", "formal"]

current_dir = os.path.dirname(os.path.abspath(__file__))
test_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/test.csv"))
predictions = [classifier(text, candidate_labels, multi_label=False) for text in test_df["text"]]

y_pred = []
y_scores = []  
for pred in predictions:
    if pred["labels"][0] == "formal":
        y_pred.append(1)
        y_scores.append(pred["scores"][0])  
    else:
        y_pred.append(0)
        y_scores.append(1 - pred["scores"][0])  

y_true = test_df["label"].tolist()

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
auc_roc = roc_auc_score(y_true, y_scores)

print(f"F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
print(f"AUC-ROC: {auc_roc:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

RocCurveDisplay.from_predictions(y_true, y_scores)
plt.title('ROC Curve')
plt.savefig('deberta_roc_curve.png', bbox_inches='tight')
plt.close()

results_df = test_df.copy()
results_df["prediction"] = y_pred
results_df["confidence_formal"] = [pred["scores"][0] for pred in predictions]
results_df["confidence_informal"] = [pred["scores"][1] for pred in predictions]
results_df.to_csv("deberta_zero_shot_predictions.csv", index=False)