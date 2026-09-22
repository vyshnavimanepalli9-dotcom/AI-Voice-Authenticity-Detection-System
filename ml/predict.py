import os
import sys
import joblib
import numpy as np
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Robust project/model paths
# ---------------------------------------------------------------------------

CURRENT_FILE = Path(__file__).resolve()
ML_DIR = CURRENT_FILE.parent
BACKEND_DIR = ML_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

MODEL_LOCATIONS = [
    BACKEND_DIR / "models",
    PROJECT_DIR / "models",
]

sys.path.insert(0, str(BACKEND_DIR))

from ml.feature_extraction import extract_features_from_audio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

def _find_model_files():
    """Find a complete model + scaler pair."""

    checked = []

    for model_dir in MODEL_LOCATIONS:
        model_path = model_dir / "voice_authenticity_rf.joblib"
        scaler_path = model_dir / "scaler.joblib"

        checked.append(str(model_dir))

        if model_path.is_file() and scaler_path.is_file():
            return model_path, scaler_path

    logger.warning(
        "Trained model files were not found. Checked: %s",
        ", ".join(checked)
    )

    return None, None


def load_trained_model():
    """
    Load trained Random Forest model and scaler.
    """

    model_path, scaler_path = _find_model_files()

    if model_path is None or scaler_path is None:
        return None, None, False

    try:
        logger.info("Loading trained model: %s", model_path)
        logger.info("Loading scaler: %s", scaler_path)

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "Loaded model does not provide predict_proba()."
            )

        if not hasattr(scaler, "transform"):
            raise TypeError(
                "Loaded scaler does not provide transform()."
            )

        logger.info(
            "Trained model loaded successfully: %s",
            type(model).__name__
        )

        return model, scaler, True

    except Exception:
        logger.exception(
            "Failed to load trained model from %s",
            model_path
        )

        return None, None, False


# ---------------------------------------------------------------------------
# ACOUSTIC SUPPORT SCORE
# ---------------------------------------------------------------------------

def calculate_acoustic_support(summary):
    """
    Calculate a conservative acoustic support score.

    IMPORTANT:
    This is NOT treated as an independent AI detector.

    It only measures whether some acoustic characteristics are
    consistent with synthetic / heavily processed speech.

    0.0 = little synthetic-style acoustic evidence
    1.0 = stronger synthetic-style acoustic evidence

    The score is intentionally conservative because acoustic
    characteristics can naturally occur in genuine human speech.
    """

    f0_var = summary.get("pitch_variation_index", 0.12)
    f0_std = summary.get("pitch_std_hz", 18.0)

    mfcc_var = summary.get("mfcc_variance", 15.0)
    delta_std = summary.get("delta_mfcc_std", 1.5)

    spectral_flux = summary.get("spectral_flux", 0.1)
    rolloff = summary.get("spectral_rolloff_hz", 3500)

    # Start neutral.
    evidence = 0.50

    # ------------------------------------------------------------------
    # 1. Pitch dynamics
    # ------------------------------------------------------------------
    # Only consider unusually low variation as weak supporting evidence.
    # Do NOT strongly penalize normal human pitch variation.
    if f0_var < 0.08 and f0_std < 15:
        evidence += 0.08
    elif f0_var > 0.25 or f0_std > 45:
        evidence -= 0.05

    # ------------------------------------------------------------------
    # 2. MFCC / timbral variation
    # ------------------------------------------------------------------
    if mfcc_var < 10:
        evidence += 0.06
    elif mfcc_var > 30:
        evidence -= 0.05

    # ------------------------------------------------------------------
    # 3. Delta-MFCC dynamics
    # ------------------------------------------------------------------
    if delta_std < 0.8:
        evidence += 0.05
    elif delta_std > 2.5:
        evidence -= 0.04

    # ------------------------------------------------------------------
    # 4. Spectral flux
    # ------------------------------------------------------------------
    # IMPORTANT:
    # The old implementation used < 1.2, which was far too broad.
    # Typical values in this implementation can be much smaller.
    #
    # Therefore only extremely low values provide weak evidence.
    if spectral_flux < 0.015:
        evidence += 0.05
    elif spectral_flux > 0.15:
        evidence -= 0.04

    # ------------------------------------------------------------------
    # 5. Spectral rolloff
    # ------------------------------------------------------------------
    # Do NOT assume <4800 Hz means AI.
    # Human speech can naturally have a lower rolloff.
    #
    # Only extremely restricted spectral bandwidth is weak evidence.
    if rolloff < 1800:
        evidence += 0.05
    elif rolloff > 5000:
        evidence -= 0.03

    return float(np.clip(evidence, 0.20, 0.80))


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

def predict_voice_authenticity(y, sr):
    """
    Main voice authenticity prediction pipeline.

    Combines:
        1. Trained Random Forest probability
        2. Conservative acoustic supporting evidence

    The acoustic score is intentionally given less influence than
    the trained ML model.
    """

    # ------------------------------------------------------------------
    # 1. Extract features
    # ------------------------------------------------------------------

    extracted = extract_features_from_audio(y, sr)

    feat_vec = extracted["feature_vector"].reshape(1, -1)

    summary = extracted["feature_summary"]

    signal_metrics = extracted["signal_metrics"]

    # ------------------------------------------------------------------
    # 2. Acoustic supporting evidence
    # ------------------------------------------------------------------

    acoustic_support = calculate_acoustic_support(summary)

    # ------------------------------------------------------------------
    # 3. Load trained model
    # ------------------------------------------------------------------

    model, scaler, is_trained = load_trained_model()

    ml_ai_prob = None

    if is_trained:

        try:

            # ----------------------------------------------------------
            # Scale feature vector
            # ----------------------------------------------------------

            scaled_vec = scaler.transform(feat_vec)

            # ----------------------------------------------------------
            # Random Forest prediction
            # ----------------------------------------------------------

            probabilities = model.predict_proba(scaled_vec)[0]

            classes = list(model.classes_)

            # ----------------------------------------------------------
            # IMPORTANT:
            # Current system assumes class 0 = AI / Fake.
            #
            # If your training dataset uses the opposite mapping,
            # this must be changed.
            # ----------------------------------------------------------

            if 0 in classes:

                fake_idx = classes.index(0)

                ml_ai_prob = float(
                    probabilities[fake_idx]
                )

            else:

                logger.warning(
                    "Class 0 not found in model classes: %s",
                    classes
                )

                ml_ai_prob = 0.5

            # ----------------------------------------------------------
            # Conservative ensemble
            #
            # ML = 80%
            # Acoustic = 20%
            #
            # This prevents a manually selected acoustic rule from
            # overpowering the trained classifier.
            # ----------------------------------------------------------

            final_ai_score = (
                0.80 * ml_ai_prob
                +
                0.20 * acoustic_support
            )

            model_type = (
                f"{type(model).__name__} "
                "+ Acoustic Evidence (Trained Model)"
            )

            demo_mode = False

            status_note = (
                "Prediction combines the trained Random Forest "
                "classifier with conservative acoustic supporting "
                "evidence."
            )

        except Exception:

            logger.exception(
                "Error during ML inference; "
                "falling back to acoustic evaluation."
            )

            final_ai_score = acoustic_support

            model_type = (
                "Acoustic Evidence Evaluator (Fallback Mode)"
            )

            demo_mode = True

            status_note = (
                "ML inference was unavailable. "
                "Only conservative acoustic evidence was used."
            )

    else:

        final_ai_score = acoustic_support

        model_type = (
            "Acoustic Evidence Evaluator (Fallback Mode)"
        )

        demo_mode = True

        status_note = (
            "Trained model unavailable. "
            "Only conservative acoustic evidence was used."
        )

    # ------------------------------------------------------------------
    # 4. Clamp final score
    # ------------------------------------------------------------------

    final_ai_score = float(
        np.clip(
            final_ai_score,
            0.05,
            0.95
        )
    )

    # ------------------------------------------------------------------
    # 5. Decision threshold
    # ------------------------------------------------------------------

    # Keep threshold at 0.50 for now.
    #
    # IMPORTANT:
    # This is not a calibrated probability unless the model was
    # properly calibrated on a representative validation set.

    AI_THRESHOLD = 0.50

    is_ai = final_ai_score >= AI_THRESHOLD

    is_human = not is_ai

    label = (
        "AI-GENERATED VOICE"
        if is_ai
        else
        "REAL HUMAN VOICE"
    )

    confidence = round(
        (
            final_ai_score
            if is_ai
            else
            (1.0 - final_ai_score)
        ) * 100,
        1
    )

    # ------------------------------------------------------------------
    # 6. XAI
    # ------------------------------------------------------------------

    xai_explanation = _generate_xai_explanations(
        summary=summary,
        is_human=is_human,
        confidence=confidence,
        demo_mode=demo_mode,
        ml_ai_prob=ml_ai_prob,
        acoustic_support=acoustic_support
    )

    # ------------------------------------------------------------------
    # 7. Return
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------

def _generate_xai_explanations(
    summary,
    is_human,
    confidence,
    demo_mode,
    ml_ai_prob=None,
    acoustic_support=0.50
):
    """
    Generate evidence-based explanation.

    Avoids claiming that a particular generator such as ElevenLabs,
    RVC or HiFi-GAN was responsible unless a dedicated attribution
    model has actually been trained for that task.
    """

    reasons = []

    f0_var = summary.get(
        "pitch_variation_index",
        0.12
    )

    f0_std = summary.get(
        "pitch_std_hz",
        18.0
    )

    mfcc_var = summary.get(
        "mfcc_variance",
        14.0
    )

    rolloff = summary.get(
        "spectral_rolloff_hz",
        3500
    )

    zcr = summary.get(
        "zero_crossing_rate",
        0.04
    )

    delta_std = summary.get(
        "delta_mfcc_std",
        1.2
    )

    spectral_flux = summary.get(
        "spectral_flux",
        0.1
    )

    # ------------------------------------------------------------------
    # AI-oriented explanation
    # ------------------------------------------------------------------

    if not is_human:

        if f0_var < 0.08 and f0_std < 15:

            reasons.append({
                "feature": "Pitch Contour Dynamics",
                "observation": (
                    f"Low pitch variation "
                    f"({round(f0_std, 2)} Hz standard deviation)"
                ),
                "impact": (
                    "Provides weak supporting evidence of "
                    "highly regular pitch dynamics."
                ),
                "type": "negative"
            })

        if mfcc_var < 10:

            reasons.append({
                "feature": "Spectral Timbre Variation",
                "observation": (
                    f"Low MFCC variance "
                    f"({round(mfcc_var, 2)})"
                ),
                "impact": (
                    "Provides supporting evidence of "
                    "relatively uniform spectral characteristics."
                ),
                "type": "negative"
            })

        if delta_std < 0.8:

            reasons.append({
                "feature": "Temporal Spectral Dynamics",
                "observation": (
                    f"Low delta-MFCC variation "
                    f"({round(delta_std, 3)})"
                ),
                "impact": (
                    "Indicates relatively stable frame-to-frame "
                    "spectral characteristics."
                ),
                "type": "negative"
            })

        if spectral_flux < 0.015:

            reasons.append({
                "feature": "Spectral Flux",
                "observation": (
                    f"Very low spectral flux "
                    f"({round(spectral_flux, 4)})"
                ),
                "impact": (
                    "Provides weak supporting evidence of "
                    "limited spectral change."
                ),
                "type": "negative"
            })

        if rolloff < 1800:

            reasons.append({
                "feature": "Spectral Rolloff",
                "observation": (
                    f"Low spectral rolloff "
                    f"({int(rolloff)} Hz)"
                ),
                "impact": (
                    "Indicates restricted high-frequency "
                    "spectral energy."
                ),
                "type": "negative"
            })

    # ------------------------------------------------------------------
    # Human-oriented explanation
    # ------------------------------------------------------------------

    else:

        if f0_var >= 0.08 or f0_std >= 15:

            reasons.append({
                "feature": "Pitch Contour Dynamics",
                "observation": (
                    f"Pitch variation detected "
                    f"({round(f0_std, 2)} Hz standard deviation)"
                ),
                "impact": (
                    "The recording contains measurable pitch "
                    "variation rather than an extremely rigid contour."
                ),
                "type": "positive"
            })

        if mfcc_var >= 10:

            reasons.append({
                "feature": "Spectral Timbre Variation",
                "observation": (
                    f"MFCC variance: "
                    f"{round(mfcc_var, 2)}"
                ),
                "impact": (
                    "The recording contains noticeable "
                    "spectral/timbral variation."
                ),
                "type": "positive"
            })

        if delta_std >= 0.8:

            reasons.append({
                "feature": "Temporal Spectral Dynamics",
                "observation": (
                    f"Delta-MFCC variation: "
                    f"{round(delta_std, 3)}"
                ),
                "impact": (
                    "The acoustic representation changes "
                    "across speech frames."
                ),
                "type": "positive"
            })

    # ------------------------------------------------------------------
    # If no specific evidence triggered
    # ------------------------------------------------------------------

    if not reasons:

        reasons.append({
            "feature": "Overall Acoustic Evidence",
            "observation": (
                "No single acoustic characteristic strongly "
                "dominated the analysis."
            ),
            "impact": (
                "The result should be interpreted together "
                "with the trained model score."
            ),
            "type": "neutral"
        })

    # ------------------------------------------------------------------
    # Model information
    # ------------------------------------------------------------------

    if ml_ai_prob is not None:

        model_probability_text = (
            f"Random Forest AI probability: "
            f"{round(ml_ai_prob * 100, 1)}%"
        )

    else:

        model_probability_text = (
            "Random Forest probability unavailable."
        )

    acoustic_text = (
        f"Acoustic supporting-evidence score: "
        f"{round(acoustic_support * 100, 1)}%"
    )

    if is_human:

        verdict_text = "REAL HUMAN VOICE"

    else:

        verdict_text = "AI-GENERATED VOICE"

    summary_text = (
        f"The system classified the recording as "
        f"{verdict_text} with a displayed confidence of "
        f"{confidence}%. "
        f"{model_probability_text} "
        f"{acoustic_text}"
    )

    return {
        "summary": summary_text,

        "factors": reasons,

        "disclaimer": (
            "This result is a probabilistic authenticity "
            "assessment based on acoustic features and a "
            "trained machine-learning classifier. It is not "
            "proof of human or AI origin and should not be "
            "treated as legal or forensic certainty."
        )
    }