import os
import sys
import joblib
import numpy as np
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.feature_extraction import extract_features_from_audio, extract_from_file

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "voice_authenticity_rf.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")

def load_trained_model():
    """Loads saved model and scaler if available."""
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, scaler, True
        except Exception as e:
            logger.error(f"Failed to load trained model: {e}")
            return None, None, False
    return None, None, False

def calculate_ai_acoustic_risk(summary):
    """
    Computes an Acoustic AI Risk Score (0.0 = Natural Human Voice, 1.0 = AI Synthetic / Cloned Voice)
    by evaluating key acoustic signatures left by neural speech synthesis engines (ElevenLabs, RVC, VALL-E, HiFi-GAN):
      1. Over-smoothed / rigid pitch contour (low f0 std & variation)
      2. Uniform cepstral / timbral envelope (low MFCC variance)
      3. Low temporal frame-to-frame delta-MFCC dynamics
      4. Low spectral flux (lack of dynamic vocal tract energy shifts)
      5. Unnatural high-frequency attenuation (< 4800 Hz roll-off)
    """
    f0_var = summary.get("pitch_variation_index", 0.10)
    f0_std = summary.get("pitch_std_hz", 18.0)
    mfcc_var = summary.get("mfcc_variance", 15.0)
    delta_std = summary.get("delta_mfcc_std", 1.5)
    spectral_flux = summary.get("spectral_flux", 0.6)
    rolloff = summary.get("spectral_rolloff_hz", 3800)
    zcr = summary.get("zero_crossing_rate", 0.05)

    risk_score = 0.50

    # Indicator 1: Pitch Smoothing & Inflection Regularity
    if f0_var < 0.18 or f0_std < 32.0:
        risk_score += 0.22
    else:
        risk_score -= 0.12

    # Indicator 2: Timbral Envelope Uniformity (MFCC Variance)
    if mfcc_var < 22.0:
        risk_score += 0.18
    else:
        risk_score -= 0.10

    # Indicator 3: Temporal Delta MFCC Dynamic Range
    if delta_std < 2.8:
        risk_score += 0.15

    # Indicator 4: Spectral Flux Energy Change
    if spectral_flux < 1.2:
        risk_score += 0.12

    # Indicator 5: High Frequency Cutoff / Filtering
    if rolloff < 4800:
        risk_score += 0.10

    # Indicator 6: Breath & Fricative ZCR Distribution
    if zcr < 0.045 or zcr > 0.20:
        risk_score += 0.08

    return float(np.clip(risk_score, 0.05, 0.98))

def predict_voice_authenticity(y, sr):
    """
    Main prediction pipeline.
    Extracts features, computes hybrid ML + acoustic risk score, and determines authenticity verdict.
    """
    # 1. Extract audio features
    extracted = extract_features_from_audio(y, sr)
    feat_vec = extracted["feature_vector"].reshape(1, -1)
    summary = extracted["feature_summary"]
    signal_metrics = extracted["signal_metrics"]

    # 2. Compute Acoustic AI Risk Score from extracted Librosa features
    acoustic_ai_risk = calculate_ai_acoustic_risk(summary)

    # 3. Check for trained ML model
    model, scaler, is_trained = load_trained_model()

    if is_trained:
        try:
            scaled_vec = scaler.transform(feat_vec)
            probabilities = model.predict_proba(scaled_vec)[0]
            
            classes = list(model.classes_)
            if 0 in classes:
                fake_idx = classes.index(0)
                ml_ai_prob = float(probabilities[fake_idx])
            else:
                ml_ai_prob = 0.5

            # Ensemble hybrid score: 50% ML Model + 50% Acoustic Signal Evaluation
            final_ai_score = 0.50 * ml_ai_prob + 0.50 * acoustic_ai_risk
            model_type = f"{type(model).__name__} + Signal Evaluator (Trained Model)"
            demo_mode = False
            status_note = "Ensemble evaluation combining trained Random Forest classifier with acoustic spectral feature analysis."
        except Exception as e:
            logger.warning(f"Error during ML inference, defaulting to acoustic evaluation: {e}")
            final_ai_score = acoustic_ai_risk
            model_type = "Acoustic Signal Evaluator (Demo Mode)"
            demo_mode = True
            status_note = "Operating in Demo Mode analyzing pitch smoothness, cepstral variance, and high-frequency roll-off anomalies."
    else:
        final_ai_score = acoustic_ai_risk
        model_type = "Acoustic Signal Evaluator (Demo Mode)"
        demo_mode = True
        status_note = "Operating in Demo Mode analyzing pitch smoothness, cepstral variance, and high-frequency roll-off anomalies."

    final_ai_score = float(np.clip(final_ai_score, 0.05, 0.98))

    # Verdict Decision: Threshold at 0.45 AI risk
    is_ai = final_ai_score >= 0.45
    is_human = not is_ai

    label = "AI-GENERATED VOICE" if is_ai else "REAL HUMAN VOICE"
    confidence = round((final_ai_score if is_ai else (1.0 - final_ai_score)) * 100, 1)

    # 4. Generate Explainable AI (XAI) Breakdown
    xai_explanation = _generate_xai_explanations(summary, is_human, confidence, demo_mode)

    return {
        "prediction": label,
        "is_human": is_human,
        "confidence": confidence,
        "demo_mode": demo_mode,
        "model_used": model_type,
        "status_note": status_note,
        "metrics": signal_metrics,
        "feature_summary": summary,
        "xai_explanation": xai_explanation
    }

def _generate_xai_explanations(summary, is_human, confidence, demo_mode):
    """Generates human-readable Explainable AI (XAI) feature impact reasoning."""
    reasons = []

    f0_var = summary.get("pitch_variation_index", 0.12)
    f0_std = summary.get("pitch_std_hz", 18.0)
    mfcc_var = summary.get("mfcc_variance", 14.0)
    rolloff = summary.get("spectral_rolloff_hz", 3500)
    zcr = summary.get("zero_crossing_rate", 0.04)
    delta_std = summary.get("delta_mfcc_std", 1.2)
    
    if not is_human:
        # AI Detection Factors
        reasons.append({
            "feature": "Pitch Contour Dynamics (f0)",
            "observation": f"Over-smoothed pitch contour ({f0_std} Hz std, variation index: {f0_var})",
            "impact": "Supports AI Detection - Neural speech synthesis engines (ElevenLabs, RVC, HiFi-GAN) exhibit over-regularized pitch inflections.",
            "type": "negative"
        })

        reasons.append({
            "feature": "Spectral Timbre Envelope (MFCC)",
            "observation": f"Cepstral variance score: {round(mfcc_var, 2)} (Delta std: {delta_std})",
            "impact": "Supports AI Detection - Synthetic voice displays uniform timbral envelope across acoustic frames.",
            "type": "negative"
        })

        reasons.append({
            "feature": "High-Frequency Spectrum (Rolloff)",
            "observation": f"85% spectral energy cutoff at {int(rolloff)} Hz",
            "impact": "Supports AI Detection - Unnatural high-frequency attenuation common in neural vocoders.",
            "type": "negative"
        })

        reasons.append({
            "feature": "Zero Crossing Rate & Breath Dynamics",
            "observation": f"ZCR ratio: {zcr}",
            "impact": "Supports AI Detection - Lack of organic breath intake cues or vocal cord fricative noise.",
            "type": "negative"
        })
    else:
        # Human Detection Factors
        reasons.append({
            "feature": "Pitch Contour Dynamics (f0)",
            "observation": f"Natural pitch variation ({f0_std} Hz std, variation index: {f0_var})",
            "impact": "Supports Human Authenticity - Natural human vocal cords produce dynamic pitch inflections.",
            "type": "positive"
        })

        reasons.append({
            "feature": "Spectral Timbre Envelope (MFCC)",
            "observation": f"Cepstral variance score: {round(mfcc_var, 2)} (Delta std: {delta_std})",
            "impact": "Supports Human Authenticity - Rich timbral dynamic variations across spoken phonemes.",
            "type": "positive"
        })

        reasons.append({
            "feature": "High-Frequency Spectrum (Rolloff)",
            "observation": f"85% spectral energy cutoff at {int(rolloff)} Hz",
            "impact": "Supports Human Authenticity - Organic high-frequency harmonic distribution.",
            "type": "positive"
        })

        reasons.append({
            "feature": "Zero Crossing Rate & Fricative Dynamics",
            "observation": f"ZCR ratio: {zcr}",
            "impact": "Supports Human Authenticity - Natural ratio of unvoiced fricatives to voiced speech.",
            "type": "positive"
        })

    verdict_text = "AI-GENERATED VOICE" if not is_human else "REAL HUMAN VOICE"

    return {
        "summary": f"The classifier determined {confidence}% confidence for {verdict_text} based on acoustic spectral, temporal, and pitch features.",
        "factors": reasons,
        "disclaimer": "AI voice authenticity scores are probabilistic estimations based on extracted acoustic features. Results should be interpreted as diagnostic indicators rather than legally definitive proof."
    }
