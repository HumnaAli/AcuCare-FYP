import numpy as np
import lightgbm as lgb  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# === Load your features ===
data = np.load('feature_kmeans_k10.npz')
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
class_names = ["nodules", "papule", "pustules"]

os.makedirs("results_lgbm", exist_ok=True)

# === Feature selection and scaling ===

X_train_fs = X_train
X_test_fs = X_test

scaler = StandardScaler().fit(X_train_fs)
X_train_scaled = scaler.transform(X_train_fs)
X_test_scaled = scaler.transform(X_test_fs)

# === LightGBM model ===
lgb_model = lgb.LGBMClassifier(
    n_estimators=50,
    learning_rate=0.005,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=5,
    reg_alpha=1.0,   # L1 regularization
    reg_lambda=1.0,  # L2 regularization
    random_state=42
)

lgb_model.fit(X_train_scaled, y_train)

# === Predictions and Accuracy ===
train_preds = lgb_model.predict(X_train_scaled)
test_preds = lgb_model.predict(X_test_scaled)

print("Train Accuracy:", accuracy_score(y_train, train_preds))
print("Test Accuracy:", accuracy_score(y_test, test_preds))

# === Reports ===
with open("results_lgbm/train_report_k_reduced.txt", "w") as f:
    f.write(classification_report(y_train, train_preds, target_names=class_names))
with open("results_lgbm/test_report_k_reduced.txt", "w") as f:
    f.write(classification_report(y_test, test_preds, target_names=class_names))

# === Confusion Matrix ===
plt.figure(figsize=(5,4))
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=class_names, yticklabels=class_names)
plt.title("LightGBM Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results_lgbm/conf_matrix_test_k_reduced.png")
plt.close()
