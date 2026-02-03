# evalute.py
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import load_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

X, y = load_data("data/flows.csv")

split = int(0.8 * len(X))
X_test, y_test = X[split:], y[split:]

model = load_model("model/reaper_model.h5")

y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", round(auc, 4))
