import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_classif

# --- Configuration ---
# Input file containing the full 49 FCM features
INPUT_FILE = 'feature_fmeans.npz' 
# Output file for the reduced 10 features
OUTPUT_FILE = 'feature_fmeans_k10.npz'
# The desired number of features (k=10 to match K-Means/ANOVA K-Best logic)
K_FEATURES = 10 

def reduce_and_save_features():
    """
    Loads the full feature set, applies SelectKBest (ANOVA F-score) 
    to reduce dimensionality to K_FEATURES, and saves the reduced set.
    """
    print("=" * 70)
    print(f"Starting Feature Reduction: {INPUT_FILE} -> K={K_FEATURES}")
    print("=" * 70)

    # 1. Load the full 49-feature FCM data
    try:
        data = np.load(INPUT_FILE, allow_pickle=True)
        
        # Extract features and labels from the full dataset
        X_train_full = data['X_train']
        y_train = data['y_train']
        X_test_full = data['X_test']
        y_test = data['y_test']
        
        # Load filenames if they exist (good for debugging, but not used in selection)
        train_filenames = data.get('train_filenames')
        test_filenames = data.get('test_filenames')

        print(f"Full Data Loaded. Train Shape: {X_train_full.shape}, Test Shape: {X_test_full.shape}")
        if X_train_full.shape[1] != 49:
            print(f"Warning: Expected 49 features, found {X_train_full.shape[1]}. Proceeding with selection.")
            
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_FILE}'. Please run your 'feat_f.py' script first.")
        return
    except Exception as e:
        print(f"An error occurred loading the NPZ file: {e}")
        return

    # 2. Apply SelectKBest on the *training* data
    # f_classif is the ANOVA F-test used for classification feature selection.
    print(f"Applying SelectKBest(k={K_FEATURES}) using f_classif on Training Data...")
    selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
    
    # Fit the selector only on the training data
    selector.fit(X_train_full, y_train)

    # Transform both training and testing data using the fitted selector
    X_train_reduced = selector.transform(X_train_full)
    X_test_reduced = selector.transform(X_test_full)
    
    print(f"Reduction Complete. Reduced Train Shape: {X_train_reduced.shape}")
    
    # Optional: Print the F-scores of the top 10 features (helpful for your slide 5)
    selected_indices = selector.get_support(indices=True)
    f_scores = selector.scores_[selected_indices]
    print("\nTop 10 Feature F-Scores:")
    for i, score in zip(selected_indices, f_scores):
        print(f"Index {i}: {score:.2f}")

    # 3. Save the reduced features to the new NPZ file
    np.savez(OUTPUT_FILE,
             X_train=X_train_reduced, y_train=y_train,
             X_test=X_test_reduced, y_test=y_test,
             train_filenames=train_filenames,
             test_filenames=test_filenames)

    print(f"\nSuccessfully saved reduced feature file: '{OUTPUT_FILE}'")
    print(f"Use this file in your classifier script for the K=10 FCM experiment.")
    print("=" * 70)


if __name__ == '__main__':
    # Run the feature reduction process
    reduce_and_save_features()