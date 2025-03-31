import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np

model_name = "google/flan-t5-large"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def improved_flant5_classifier(text):
    """Enhanced classification with contrastive prompting"""
    prompt = f"""Analyze this text's formality level. Choose between:
    A) Formal: Uses proper grammar, complex vocabulary, professional tone
    B) Informal: Casual language, contractions, colloquialisms
    
    Text: "{text}"
    
    The correct classification is:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        formal_inputs = tokenizer(prompt + " A) Formal", return_tensors="pt").to(device)
        informal_inputs = tokenizer(prompt + " B) Informal", return_tensors="pt").to(device)
        
        formal_score = model(**formal_inputs, labels=formal_inputs["input_ids"]).loss.item()
        informal_score = model(**informal_inputs, labels=informal_inputs["input_ids"]).loss.item()
    
    probs = np.exp(-np.array([formal_score, informal_score]))
    probs /= probs.sum()
    
    pred = 1 if probs[0] > probs[1] else 0
    confidence = max(probs[0], probs[1])
    
    return pred, confidence

current_dir = os.path.dirname(os.path.abspath(__file__))
test_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/test.csv"))
predictions = []
for i, text in enumerate(test_df["text"]):
    pred, conf = improved_flant5_classifier(text)
    predictions.append((pred, conf))

y_pred = [p[0] for p in predictions]
y_scores = [p[1] for p in predictions]
y_true = test_df["label"].tolist()

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
auc_roc = roc_auc_score(y_true, y_scores)

print(f"F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
print(f"AUC-ROC: {auc_roc:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

RocCurveDisplay.from_predictions(y_true, y_scores)
plt.title('ROC Curve')
plt.savefig('flan_t5_roc.png', bbox_inches='tight')
plt.close()

results_df = test_df.copy()
results_df["prediction"] = y_pred
results_df["confidence"] = y_scores
results_df.to_csv("flant5_predictions.csv", index=False)