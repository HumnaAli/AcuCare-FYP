import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multiclass import OneVsRestClassifier
import joblib

# === Config ===
class_names = ["nodules", "papule", "pustules"]
os.makedirs("results_rf", exist_ok=True)

# === Load Dataset ===
data = np.load('feature_kmeans_k10.npz', allow_pickle=True)
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# === Feature Selection ===

X_train_fs = X_train
X_test_fs  = X_test

# === Standardization ===
scaler = StandardScaler().fit(X_train_fs)
X_train_scaled = scaler.transform(X_train_fs)
X_test_scaled  = scaler.transform(X_test_fs)

# === Train RF with proper settings ===
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# === Predictions ===
y_pred_train = rf.predict(X_train_scaled)
y_pred_test  = rf.predict(X_test_scaled)

# === Accuracy ===
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

# === Reports ===
with open("results_rf/train_report_k_reduced.txt", "w") as f:
    f.write(classification_report(y_train, y_pred_train, target_names=class_names))
with open("results_rf/test_report_k_reduced.txt", "w") as f:
    f.write(classification_report(y_test, y_pred_test, target_names=class_names))

# === Confusion Matrix (Test) ===
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Random Forest - Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results_rf/conf_matrix_test_k_reduced.png")
plt.close()

# === ROC Curve ===
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = OneVsRestClassifier(rf).fit(X_train_scaled, y_train).predict_proba(X_test_scaled)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i, name in enumerate(class_names):
    plt.plot(fpr[i], tpr[i], label=f"{name} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Random Forest - ROC Curve (Test)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results_rf/roc_curve_test_k_reduced.png")
plt.close()

# === Save RF Pipeline ===
joblib.dump({'model': rf, 'scaler': scaler}, 'rf_pipeline_k_reduced.joblib')
print("RF model pipeline saved to rf_pipeline_k_reduced.joblib")
