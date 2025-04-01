import pandas as pd
import os
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    BartForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
test_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/test.csv"))

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

def predict_formality_with_confidence(text):
    result = classifier(
        text,
        candidate_labels=["informal", "formal"],
        hypothesis_template="This text is {}."  
    )
    pred = 1 if result['labels'][0] == "formal" else 0
    confidence = result['scores'][0] if pred == 1 else result['scores'][1]
    return pred, confidence

def batch_predict(texts):
    predictions = []
    confidences = []
    for text in texts:
        pred, conf = predict_formality_with_confidence(text)
        predictions.append(pred)
        confidences.append(conf)
    return predictions, confidences

y_pred, y_scores = batch_predict(test_df["text"].tolist())
y_true = test_df["label"].tolist()

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
auc_roc = roc_auc_score(y_true, y_scores)  

print(f"\nF1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
print(f"AUC-ROC: {auc_roc:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

RocCurveDisplay.from_predictions(y_true, y_scores)
plt.title('ROC Curve')
plt.savefig(os.path.join(current_dir, "../../results/plots/roc_curve_zsBART.png"))  
plt.close()

results_df = test_df.copy()
results_df['prediction'] = y_pred
results_df['confidence_score'] = y_scores
results_df.to_csv(os.path.join(current_dir, "../../results/bart_zero_shot_predictions.csv"), index=False)
