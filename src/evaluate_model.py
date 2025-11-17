# src/evaluate_model.py
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data_preprocessing import load_data, preprocess_and_save
import numpy as np

df = load_data()
X_scaled, y, le, scaler = preprocess_and_save(df)
model = joblib.load("models/final_model.pkl")

# Evaluate on full dataset (or split if you saved X_test)
y_pred = model.predict(X_scaled)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (All Data)")
plt.show()
