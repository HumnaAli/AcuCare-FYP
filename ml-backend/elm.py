import numpy as np
import seaborn as sns
import os
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelBinarizer, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt




feature_names = [
    "GLCM_contrast", "GLCM_correlation", "GLCM_energy", "GLCM_homogeneity"
] + [f"Gabor_{i}" for i in range(16)]
class_names = ["nodules", "papule", "pustules"]

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import numpy as np

class ELMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, hidden_neurons=50, activation='sigmoid', random_state=0, C=1e3):
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        self.activation = activation
        self.random_state = random_state
        self.C = C

    def _activate(self, X):
        H_raw = X.dot(self.w_hidden) + self.b_hidden
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-H_raw))
        elif self.activation == "relu":
            return np.maximum(0, H_raw)
        elif self.activation == "rbf":
            H = np.zeros((X.shape[0], self.hidden_neurons))
            for i in range(self.hidden_neurons):
                center = self.w_hidden[:, i]
                sigma = 1.0
                dist = np.linalg.norm(X - center, axis=1)
                H[:, i] = np.exp(-(dist ** 2) / (2 * sigma ** 2))
            return H
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        lb = LabelBinarizer()
        Y_onehot = lb.fit_transform(y)
        self.label_binarizer_ = lb

        rng = np.random.RandomState(self.random_state)
        self.input_dim_ = X.shape[1] if self.input_dim is None else self.input_dim
        self.w_hidden = rng.randn(self.input_dim_, self.hidden_neurons).astype(np.float32)
        self.b_hidden = rng.randn(self.hidden_neurons).astype(np.float32)

        H = self._activate(X)
        lambda_identity = np.eye(H.shape[1]) / self.C
        self.beta_output = np.linalg.solve(H.T @ H + lambda_identity, H.T @ Y_onehot)
        return self

    def predict_proba(self, X):
        H = self._activate(X)
        scores = H.dot(self.beta_output)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
def ensemble_predict_proba(elm_class, X, y_train, n_models=5, **elm_params):
    probas_test = []
    probas_train = []

    for seed in range(n_models):
        elm = elm_class(random_state=seed, **elm_params)
        elm.fit(X['train'], y_train)
        probas_test.append(elm.predict_proba(X['test']))
        probas_train.append(elm.predict_proba(X['train']))

    avg_proba_test = np.mean(probas_test, axis=0)
    avg_proba_train = np.mean(probas_train, axis=0)

    preds_test = np.argmax(avg_proba_test, axis=1)
    preds_train = np.argmax(avg_proba_train, axis=1)

    return preds_train, preds_test



def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", fname=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    plt.close()

def plot_roc_auc_curves(y_true, y_score, class_names, fname_prefix="roc"):
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = y_true_bin.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"results/{fname_prefix}.png")
    plt.close()


def save_model(model, scaler, selector, filename='elm_kmeans.joblib'):
    joblib.dump({'model': model, 'scaler': scaler, 'selector': selector}, filename)
    print(f"Model pipeline saved to {filename}")

def main():
    os.makedirs("results", exist_ok=True)
    data = np.load('features_kmeans.npz', allow_pickle=True)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test  = data['X_test']
    y_test  = data['y_test']
    train_filenames = data['train_filenames']
    test_filenames = data['test_filenames']

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    param_grid = {
        'hidden_neurons': [25, 50, 75, 125],
        'activation': ['sigmoid', 'relu', 'rbf'],
        'k': [10, 20, 30, 40, 50, 60],
        'C': [1e2, 1e3, 1e4]
    }

    best_score = 0
    best_params = {}
    best_model = None
    best_scaler = None
    best_selector = None
    kfold_accuracies = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k in param_grid['k']:
        for hidden in param_grid['hidden_neurons']:
            for act in param_grid['activation']:
                for C in param_grid['C']:
                    fold_scores = []
                    for train_idx, val_idx in skf.split(X_train, y_train):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]
                        selector = SelectKBest(score_func=f_classif, k=k)
                        X_tr_fs = selector.fit_transform(X_tr, y_tr)
                        X_val_fs = selector.transform(X_val)
                        scaler = StandardScaler().fit(X_tr_fs)
                        X_tr_scaled = scaler.transform(X_tr_fs)
                        X_val_scaled = scaler.transform(X_val_fs)
                        elm = ELMClassifier(input_dim=X_tr_scaled.shape[1], hidden_neurons=hidden, activation=act, random_state=42, C=C)
                        elm.fit(X_tr_scaled, y_tr)
                        preds = elm.predict(X_val_scaled)
                        score = accuracy_score(y_val, preds)
                        fold_scores.append(score)
                    mean_score = np.mean(fold_scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'k': k, 'hidden_neurons': hidden, 'activation': act, 'C': C}
                        best_selector = SelectKBest(score_func=f_classif, k=k).fit(X_train, y_train)
                        X_train_fs = best_selector.transform(X_train)
                        best_scaler = StandardScaler().fit(X_train_fs)
                        X_train_scaled = best_scaler.transform(X_train_fs)
                        best_model = ELMClassifier(input_dim=X_train_scaled.shape[1], hidden_neurons=hidden, activation=act, random_state=42, C=C)
                        best_model.fit(X_train_scaled, y_train)
                        kfold_accuracies = fold_scores.copy()

    print("\nBest CV score: {:.2f}% with params: {}".format(best_score*100, best_params))
    print("K-Fold Accuracies:", [f"{a*100:.2f}%" for a in kfold_accuracies])

    selected_idx = best_selector.get_support(indices=True)
    print(f"\nSelected {len(selected_idx)} best features (by index): {selected_idx}")
    try:
        selected_features = [feature_names[i] for i in selected_idx]
        print("\nBest features selected by SelectKBest:")
        for f in selected_features:
            print(f"  {f}")
    except:
        print("\n(Could not map feature indices to names—check if feature_names list matches feature vector length.)")

    X_train_fs = best_selector.transform(X_train)
    X_train_scaled = best_scaler.transform(X_train_fs)
    train_preds = best_model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    print("\nTraining accuracy: {:.2f}%".format(train_acc * 100))

    train_cm = confusion_matrix(y_train, train_preds)
    plot_confusion_matrix(train_cm, class_names, title="Confusion Matrix (Train)", fname="results/conf_matrix_train_kmeans.png")
    with open("results/train_report_kmeans.txt", "w") as f:
        f.write(classification_report(y_train, train_preds, target_names=class_names))


    train_report_dict = classification_report(y_train, train_preds, target_names=class_names, output_dict=True)
    print("\nPer-class Training Accuracy (Recall):")
    for cls in class_names:
        acc = train_report_dict[cls]["recall"] * 100
        print(f"{cls}: {acc:.2f}%")

    X_test_fs = best_selector.transform(X_test)
    X_test_scaled = best_scaler.transform(X_test_fs)
    test_preds = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    print("\nTest accuracy: {:.2f}%".format(test_acc * 100))

    # NOW it's safe to do this:
    test_report_dict = classification_report(y_test, test_preds, target_names=class_names, output_dict=True)
    print("\nPer-class Test Accuracy (Recall):")
    for cls in class_names:
        acc = test_report_dict[cls]["recall"] * 100
        print(f"{cls}: {acc:.2f}%")



    print("Classification Report (Train):")
    print(classification_report(y_train, train_preds, target_names=class_names))

    X_test_fs = best_selector.transform(X_test)
    X_test_scaled = best_scaler.transform(X_test_fs)
    test_preds = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    print("\nTest accuracy: {:.2f}%".format(test_acc * 100))

    test_proba = best_model.predict_proba(X_test_scaled)
    plot_roc_auc_curves(y_test, test_proba, class_names, fname_prefix="roc_kmeans_elm")

    test_cm = confusion_matrix(y_test, test_preds)
    plot_confusion_matrix(test_cm, class_names, title="Confusion Matrix (Test)", fname="results/conf_matrix_test_kmeans.png")
    with open("results/test_report_kmeans.txt", "w") as f:
        f.write(classification_report(y_test, test_preds, target_names=class_names))
        
    print("Classification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=class_names))

    print("\n=== TP, TN, FP, FN per class ===")
    labels = [0, 1, 2]  # nodules, papule, pustules

    for i, cls_name in enumerate(class_names):
        TP = test_cm[i, i]
        FN = test_cm[i, :].sum() - TP
        FP = test_cm[:, i].sum() - TP
        TN = test_cm.sum() - (TP + FN + FP)

        print(f"\nClass: {cls_name}")
        print(f"  TP: {TP}")
        print(f"  TN: {TN}")
        print(f"  FP: {FP}")
        print(f"  FN: {FN}")


    # === Add this block after test_preds is safely created ===
    test_report_dict = classification_report(y_test, test_preds, target_names=class_names, output_dict=True)
    print("\nPer-class Test Accuracy (Recall):")
    for cls in class_names:
        acc = test_report_dict[cls]["recall"] * 100
        print(f"{cls}: {acc:.2f}%")




    test_cm = confusion_matrix(y_test, test_preds)
    plot_confusion_matrix(test_cm, class_names, title="Confusion Matrix (Test)", fname="results/conf_matrix_test_kmeans.png")
    with open("results/test_report_kmeans.txt", "w") as f:
        f.write(classification_report(y_test, test_preds, target_names=class_names))

    print("Classification Report (Test):")
    print(classification_report(y_test, test_preds, target_names=class_names))

    print("\n--- Misclassified Nodules (label=0): ---")
    for idx, (true, pred) in enumerate(zip(y_test, test_preds)):
        if true == 0 and pred != 0:
            print(f"Index {idx}: {test_filenames[idx]} (predicted as {class_names[pred]})")

    print("\n--- Misclassified Papules (label=1): ---")
    for idx, (true, pred) in enumerate(zip(y_test, test_preds)):
        if true == 1 and pred != 1:
            print(f"Index {idx}: {test_filenames[idx]} (predicted as {class_names[pred]})")

    save_model(best_model, best_scaler, best_selector, filename='elm_kmeans.joblib')
    elm_params = {
    "input_dim": X_train_scaled.shape[1],
    "hidden_neurons": 25,
    "activation": 'relu',
    "C": 100.0
}

    ensemble_train_preds, ensemble_test_preds = ensemble_predict_proba(
        ELMClassifier,
        X={'train': X_train_scaled, 'test': X_test_scaled},
        y_train=y_train,
        n_models=15,
        **elm_params
    )

    ensemble_train_acc = accuracy_score(y_train, ensemble_train_preds)
    ensemble_test_acc = accuracy_score(y_test, ensemble_test_preds)

    print(f"\nEnsemble Training Accuracy: {ensemble_train_acc * 100:.2f}%")
    print(f"Ensemble Test Accuracy: {ensemble_test_acc * 100:.2f}%")



if __name__ == "__main__":
    main()