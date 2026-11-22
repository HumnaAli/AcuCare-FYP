import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.multiclass import OneVsRestClassifier
import joblib

# === Config ===
class_names = ["nodules", "papule", "pustules"]
os.makedirs("results_knn", exist_ok=True)

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

# === Train KNN (regularized) ===
knn = KNeighborsClassifier(
    n_neighbors=7,
    weights='uniform',
    p=2  # Euclidean distance
)
knn.fit(X_train_scaled, y_train)

# === Predictions ===
y_pred_train = knn.predict(X_train_scaled)
y_pred_test  = knn.predict(X_test_scaled)

# === Accuracy ===
print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

# === Reports ===
with open("results_knn/train_report_k_reduced.txt", "w") as f:
    f.write(classification_report(y_train, y_pred_train, target_names=class_names))
with open("results_knn/test_report_k_reduced.txt", "w") as f:
    f.write(classification_report(y_test, y_pred_test, target_names=class_names))

# === Confusion Matrix (Test) ===
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("KNN - Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results_knn/conf_matrix_test_k_Reduced.png")
plt.close()

# === ROC Curve ===
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = OneVsRestClassifier(knn).fit(X_train_scaled, y_train).predict_proba(X_test_scaled)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i, name in enumerate(class_names):
    plt.plot(fpr[i], tpr[i], label=f"{name} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("KNN - ROC Curve (Test)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results_knn/roc_curve_test_k_Reduced.png")
plt.close()

# === Save KNN Model Pipeline ===
joblib.dump({'model': knn, 'scaler': scaler}, 'knn_pipeline_k_reduced.joblib')
print("KNN model pipeline saved to knn_pipelin_k_reduced.joblib")
