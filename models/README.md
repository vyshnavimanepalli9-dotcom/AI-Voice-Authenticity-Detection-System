# Models Directory

Trained Scikit-Learn models (`.joblib` format) and scaler models are stored in this directory.

- `voice_authenticity_rf.joblib`: Primary Random Forest audio authenticity model.
- `scaler.joblib`: StandardScaler for extracted audio features.

If no model is found in this directory, the application gracefully operates in **DEMO MODE** using calibrated acoustic signal heuristics.
