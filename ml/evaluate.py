import os
import sys
import joblib
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "voice_authenticity_rf.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")

def evaluate_model_on_data(X_test, y_test):
    """Evaluates persisted Random Forest model against a test matrix X_test, y_test."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model binary not found in models/ directory. Run 'python ml/train.py' first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["AI-Generated", "Real Human"], output_dict=True)

    metrics = {
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    return metrics

if __name__ == "__main__":
    from ml.train import generate_synthetic_benchmark_dataset
    print("Running evaluation script on test dataset...")
    X, y = generate_synthetic_benchmark_dataset(num_samples_per_class=50)
    try:
        results = evaluate_model_on_data(X, y)
        print("\n--- EVALUATION SUMMARY ---")
        print(f"Accuracy: {results['accuracy'] * 100}%")
        print(f"ROC-AUC:  {results['roc_auc']}")
        print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
