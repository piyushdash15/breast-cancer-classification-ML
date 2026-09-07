# -*- coding: utf-8 -*-
"""
Model Test Script for Breast Cancer Classifier
Module: model_test.py

Description:
Randomly selects 20 patient cases from data.csv, evaluates each through
the trained model (breast_cancer_model.pkl), and outputs a detailed diagnostic
comparison showing whether the model was correct or incorrect on each case.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Ensure UTF-8 output on Windows consoles if supported
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "breast_cancer_model.pkl"

def run_model_test(num_samples: int = 20, random_seed: int = None):
    print("=" * 88)
    print(f"BREAST CANCER MODEL RANDOM SAMPLING EVALUATION ({num_samples} CASES)")
    print("=" * 88)

    # 1. Verify existence of model and dataset
    if not MODEL_PATH.exists():
        print(f"[!] Error: Model file '{MODEL_PATH.name}' not found!")
        print("Please ensure 'breast_cancer_model.pkl' is in the project directory.")
        sys.exit(1)

    if not DATA_PATH.exists():
        print(f"[!] Error: Dataset file '{DATA_PATH.name}' not found!")
        print("Please ensure 'data.csv' is in the project directory.")
        sys.exit(1)

    # 2. Load trained pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"[OK] Loaded model artifact: '{MODEL_PATH.name}'")
    except Exception as e:
        print(f"[!] Error loading model: {e}")
        sys.exit(1)

    # 3. Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Loaded dataset: '{DATA_PATH.name}' ({len(df)} total cases)")

    # 4. Standardize ground truth: 1 = Malignant (M), 0 = Benign (B)
    diag_str = df["diagnosis"].astype(str).str.strip().str.upper()
    if set(diag_str.unique()).issubset({"M", "B"}):
        y_true_binary = diag_str.map({"M": 1, "B": 0}).astype(int)
    else:
        # Fallback for numerical labels 1 / 0
        y_true_binary = pd.to_numeric(df["diagnosis"], errors="coerce").fillna(0).astype(int)

    # Identify the 30 features expected by the pipeline
    drop_cols = [c for c in ["id", "diagnosis", "Unnamed: 32"] if c in df.columns]
    feature_df = df.drop(columns=drop_cols)

    # 5. Randomly sample cases
    sample_kwargs = {"n": min(num_samples, len(df))}
    if random_seed is not None:
        sample_kwargs["random_state"] = random_seed
        print(f"[*] Random Seed: {random_seed} (Reproducible sample)")
    else:
        print("[*] Random Seed: None (Fresh random sample each run)")

    sample_df = df.sample(**sample_kwargs)
    sampled_indices = sample_df.index
    sample_features = feature_df.loc[sampled_indices]
    sample_true_binary = y_true_binary.loc[sampled_indices]
    sample_ids = sample_df["id"] if "id" in sample_df.columns else sampled_indices

    # 6. Run predictions and probabilities
    preds = pipeline.predict(sample_features)
    probs = pipeline.predict_proba(sample_features)

    # 7. Print formatted case-by-case results table
    print("\n" + "-" * 88)
    header = f"{'#':<3} | {'Patient ID':<12} | {'Ground Truth':<12} | {'Prediction':<12} | {'Confidence':<10} | {'Status':<14}"
    print(header)
    print("-" * 88)

    correct_count = 0
    benign_correct = 0
    benign_total = 0
    malignant_correct = 0
    malignant_total = 0

    for i, idx in enumerate(sampled_indices, 1):
        pid = str(sample_ids.loc[idx])
        true_val = int(sample_true_binary.loc[idx])
        pred_val = int(preds[i - 1])
        prob_dist = probs[i - 1]

        # Diagnosis string representations
        true_label = "Malignant" if true_val == 1 else "Benign"
        pred_label = "Malignant" if pred_val == 1 else "Benign"

        # Model confidence for the predicted class
        confidence = prob_dist[pred_val] * 100.0

        # Verification check
        is_correct = (true_val == pred_val)
        if is_correct:
            status_str = "CORRECT [OK]"
            correct_count += 1
        else:
            status_str = "MISCLASSIFIED [X]"

        # Sub-metrics
        if true_val == 0:
            benign_total += 1
            if is_correct:
                benign_correct += 1
        else:
            malignant_total += 1
            if is_correct:
                malignant_correct += 1

        row_str = f"{i:<3} | {pid:<12} | {true_label:<12} | {pred_label:<12} | {confidence:>8.2f}% | {status_str:<14}"
        print(row_str)

    # 8. Summary Statistics
    accuracy_pct = (correct_count / len(sampled_indices)) * 100.0
    print("-" * 88)
    print("\n" + "=" * 88)
    print("EVALUATION SUMMARY")
    print("=" * 88)
    print(f"Total Cases Evaluated   : {len(sampled_indices)}")
    print(f"Correct Predictions     : {correct_count} / {len(sampled_indices)} ({accuracy_pct:.1f}%)")
    print(f"Misclassifications      : {len(sampled_indices) - correct_count} / {len(sampled_indices)}")

    if benign_total > 0:
        b_acc = (benign_correct / benign_total) * 100.0
        print(f"Benign Cases Accuracy   : {benign_correct} / {benign_total} ({b_acc:.1f}%)")
    if malignant_total > 0:
        m_acc = (malignant_correct / malignant_total) * 100.0
        print(f"Malignant Cases Accuracy: {malignant_correct} / {malignant_total} ({m_acc:.1f}%)")
    print("=" * 88)

if __name__ == "__main__":
    # If a seed argument is passed (e.g., `python model_test.py 42`), use it
    seed_arg = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    run_model_test(num_samples=20, random_seed=seed_arg)
