import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import  f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# === Config ===
class_names = ["nodules", "papule", "pustules"]
os.makedirs("results_svm", exist_ok=True)

# === Load Dataset ===
data = np.load('feature_kmeans_k10.npz', allow_pickle=True)
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

# === Pipeline ===
pipeline = Pipeline([
    
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])

# === Param Grid ===
param_grid = {
    'svm__C': [0.01, 0.1, 1.0],
    'svm__gamma': [0.01, 0.05, 'scale'],
    'svm__kernel': ['rbf']
}

# === Grid Search ===
grid = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)

# === Evaluate on Train/Test ===
best_model = grid.best_estimator_
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

from sklearn.model_selection import cross_val_score

# Optional: Set cv folds for post-check
cv_scores = cross_val_score(best_model, X_train, y_train, cv=10)

print("Post-fit Cross-val Accuracy: %.2f%% ± %.2f%%" % (cv_scores.mean() * 100, cv_scores.std() * 100))


print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test))




# === Save Reports ===
with open("results_svm/train_report_k_reduced.txt", "w") as f:
    f.write(classification_report(y_train, y_pred_train, target_names=class_names))
with open("results_svm/test_report_k_reduced.txt", "w") as f:
    f.write(classification_report(y_test, y_pred_test, target_names=class_names))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("SVM - Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results_svm/conf_matrix_test_k_reduced.png")
plt.close()

# === ROC Curve ===
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = best_model.predict_proba(X_test)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i, name in enumerate(class_names):
    plt.plot(fpr[i], tpr[i], label=f"{name} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("SVM - ROC Curve (Test)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results_svm/roc_curve_test_k_reduced.png")
plt.close()

# === Save Model ===
joblib.dump(best_model, 'svm_pipeline_k_reduced.joblib')
print("SVM model pipeline saved to svm_pipeline.joblib")
