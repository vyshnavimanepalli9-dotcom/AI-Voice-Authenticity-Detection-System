import os
import sys
import glob
import argparse
import numpy as np
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from ml.feature_extraction import extract_from_file

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
MODEL_PATH = os.path.join(MODELS_DIR, "voice_authenticity_rf.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

def generate_synthetic_benchmark_dataset(num_samples_per_class=150):
    """
    Generates a realistic synthetic benchmark feature dataset (77 acoustic features)
    with controlled distributions representing Real Human speech vs AI-Generated speech.
    Used for initial model initialization when local raw audio folders are empty.
    """
    print(f"Generating synthetic feature dataset ({num_samples_per_class * 2} samples)...")
    np.random.seed(42)

    X, y = [], []

    for _ in range(num_samples_per_class):
        # --- Class 1: REAL HUMAN VOICE ---
        # Real human speech features: higher MFCC variability, natural pitch variation, dynamic pitch contour
        mfcc_mean_real = np.random.normal(loc=-150, scale=30, size=20)
        mfcc_std_real = np.random.normal(loc=25, scale=5, size=20)
        mel_real = [np.random.normal(-40, 5), np.random.normal(15, 3)]
        cent_real = [np.random.normal(2200, 300), np.random.normal(800, 100)]
        bw_real = [np.random.normal(2100, 250), np.random.normal(400, 50)]
        roll_real = [np.random.normal(4500, 600), np.random.normal(1200, 200)]
        zcr_real = [np.random.uniform(0.04, 0.12), np.random.uniform(0.02, 0.05)]
        chroma_mean_real = np.random.uniform(0.2, 0.6, size=12)
        chroma_std_real = np.random.uniform(0.1, 0.3, size=12)
        f0_stats_real = [np.random.normal(160, 25), np.random.normal(35, 10), np.random.uniform(0.12, 0.35)]

        delta_std_real = np.random.normal(3.5, 0.5)
        spectral_flux_real = np.random.normal(1.5, 0.3)

        feat_real = np.hstack([
            mfcc_mean_real, mfcc_std_real, mel_real, cent_real, bw_real, roll_real, zcr_real,
            chroma_mean_real, chroma_std_real, f0_stats_real, delta_std_real, spectral_flux_real
        ])
        X.append(feat_real)
        y.append(1) # 1 = Real Human Voice

        # --- Class 0: AI-GENERATED / SYNTHETIC VOICE ---
        # AI speech features: over-smoothed pitch contours, lower MFCC variability, high-freq roll-off artifacts
        mfcc_mean_synth = np.random.normal(loc=-120, scale=15, size=20)
        mfcc_std_synth = np.random.normal(loc=12, scale=3, size=20) # over-smoothed
        mel_synth = [np.random.normal(-35, 3), np.random.normal(8, 2)]
        cent_synth = [np.random.normal(1800, 150), np.random.normal(400, 50)]
        bw_synth = [np.random.normal(1700, 150), np.random.normal(250, 30)]
        roll_synth = [np.random.normal(3200, 300), np.random.normal(600, 100)]
        zcr_synth = [np.random.uniform(0.02, 0.06), np.random.uniform(0.005, 0.02)]
        chroma_mean_synth = np.random.uniform(0.3, 0.5, size=12)
        chroma_std_synth = np.random.uniform(0.05, 0.15, size=12)
        f0_stats_synth = [np.random.normal(150, 10), np.random.normal(10, 3), np.random.uniform(0.01, 0.08)] # Low pitch variance
        delta_std_synth = np.random.normal(1.1, 0.2)
        spectral_flux_synth = np.random.normal(0.4, 0.1)

        feat_synth = np.hstack([
            mfcc_mean_synth, mfcc_std_synth, mel_synth, cent_synth, bw_synth, roll_synth, zcr_synth,
            chroma_mean_synth, chroma_std_synth, f0_stats_synth, delta_std_synth, spectral_flux_synth
        ])
        X.append(feat_synth)
        y.append(0) # 0 = AI-Generated Voice


    return np.array(X), np.array(y)

def train_model(X, y):
    """Trains a Random Forest Classifier and saves model & scaler."""
    print("Partitioning dataset into Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Fitting StandardScaler on audio feature matrix...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Classifier (n_estimators=100)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Evaluation
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    print("\n--- MODEL EVALUATION METRICS ---")
    print(classification_report(y_test, y_pred, target_names=["AI-Generated (0)", "Real Human (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # Persistence
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nModel successfully persisted to: {MODEL_PATH}")
    print(f"Scaler successfully persisted to: {SCALER_PATH}")
    return clf, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Voice Authenticity Classifier")
    parser.add_argument("--generate-synthetic", action="store_true", help="Force synthetic dataset generation")
    args = parser.parse_args()

    real_files = glob.glob(os.path.join(DATASETS_DIR, "real", "*.wav")) + glob.glob(os.path.join(DATASETS_DIR, "real", "*.mp3"))
    synth_files = glob.glob(os.path.join(DATASETS_DIR, "synthetic", "*.wav")) + glob.glob(os.path.join(DATASETS_DIR, "synthetic", "*.mp3"))

    if len(real_files) > 0 and len(synth_files) > 0 and not args.generate_synthetic:
        print(f"Found {len(real_files)} real voice files and {len(synth_files)} synthetic voice files.")
        X, y = [], []
        for f in real_files:
            try:
                res = extract_from_file(f)
                X.append(res["feature_vector"])
                y.append(1)
            except Exception as e:
                print(f"Error extracting {f}: {e}")

        for f in synth_files:
            try:
                res = extract_from_file(f)
                X.append(res["feature_vector"])
                y.append(0)
            except Exception as e:
                print(f"Error extracting {f}: {e}")

        X, y = np.array(X), np.array(y)
    else:
        X, y = generate_synthetic_benchmark_dataset(num_samples_per_class=150)

    train_model(X, y)
