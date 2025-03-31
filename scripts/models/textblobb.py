from textblob import TextBlob
import os
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support, 
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
import matplotlib.pyplot as plt
import nltk

nltk.download('punkt')  
nltk.download('averaged_perceptron_tagger')

def textblob_formality_score_with_confidence(text):
    """Returns both formality prediction (0/1) and confidence score (0-1)"""
    blob = TextBlob(text)
    
    contractions = {"'s", "'re", "'ll", "n't", "'ve"}
    has_contractions = any(c in text.lower() for c in contractions)
    
    avg_word_len = sum(len(word) for word in blob.words) / max(1, len(blob.words))
    
    informal_verbs = {"gonna", "wanna", "gotta", "shoulda", "coulda"}
    has_informal_verbs = any(tag[1] == 'VB' and tag[0].lower() in informal_verbs 
                          for tag in blob.tags)
    
    score = 0.7 * (not has_contractions) + 0.2 * (avg_word_len > 4) + 0.1 * (not has_informal_verbs)
    
    prediction = 1 if score > 0.5 else 0
    confidence = abs(score - 0.5) * 2  
    
    return prediction, confidence

current_dir = os.path.dirname(os.path.abspath(__file__))
test_df = pd.read_csv(os.path.join(current_dir, "../../data/processed/test.csv"))

predictions = [textblob_formality_score_with_confidence(text) for text in test_df["text"]]
y_pred = [pred[0] for pred in predictions]  
y_scores = [pred[1] for pred in predictions]  
y_true = test_df["label"].tolist()

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
auc_roc = roc_auc_score(y_true, y_scores)  

print(f"F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
print(f"AUC-ROC: {auc_roc:.3f}")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

RocCurveDisplay.from_predictions(y_true, y_scores)
plt.title('ROC Curve')
plt.savefig('textblob_roc_curve.png')
plt.close()

results_df = test_df.copy()
results_df["prediction"] = y_pred
results_df["confidence"] = y_scores
results_df.to_csv("textblob_predictions.csv", index=False)