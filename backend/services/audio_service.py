import time
import logging

import librosa
import numpy as np

from ml.preprocessing import validate_audio_signal, normalize_signal
from ml.predict import predict_voice_authenticity


logger = logging.getLogger(__name__)


def process_and_analyze_audio(file_path):
    """
    Loads an audio file, validates the signal, normalizes it,
    runs the voice authenticity classifier, and prepares
    waveform and audio metadata for the API response.
    """

    start_time = time.time()
    target_sr = 22050

    # ---------------------------------------------------------
    # 1. Load audio
    # ---------------------------------------------------------
    try:
        y, sr = librosa.load(
            file_path,
            sr=target_sr,
            mono=True
        )

        # Limit analysis duration for deployment stability
        MAX_DURATION_SECONDS = 30

        max_samples = int(
            MAX_DURATION_SECONDS * sr
        )

        if len(y) > max_samples:
            y = y[:max_samples]

    except Exception as e:
        logger.error(
            "Librosa audio loading error: %s",
            e
        )

        raise ValueError(
            "Failed to decode audio file. "
            "Ensure the file format is supported. "
            f"Error: {str(e)}"
        )

    # ---------------------------------------------------------
    # 2. Validate audio signal
    # ---------------------------------------------------------
    try:
        is_valid, warning_msg = validate_audio_signal(
            y,
            sr
        )

        if not is_valid:
            raise ValueError(warning_msg)

    except ValueError:
        raise

    except Exception as e:
        logger.error(
            "Audio validation error: %s",
            e
        )

        raise ValueError(
            f"Audio validation failed: {str(e)}"
        )

    # ---------------------------------------------------------
    # 3. Normalize audio
    # ---------------------------------------------------------
    try:
        y = normalize_signal(y)

    except Exception as e:
        logger.error(
            "Audio normalization error: %s",
            e
        )

        raise ValueError(
            f"Audio normalization failed: {str(e)}"
        )

    # ---------------------------------------------------------
    # 4. Generate waveform data
    # ---------------------------------------------------------
    waveform_points = generate_waveform_points(
        y,
        num_points=100
    )

    # ---------------------------------------------------------
    # 5. Run ML authenticity classifier
    # ---------------------------------------------------------
    try:
        prediction_result = predict_voice_authenticity(
            y,
            sr
        )

    except Exception as e:
        logger.exception(
            "Voice authenticity prediction failed"
        )

        raise RuntimeError(
            f"Voice authenticity analysis failed: {str(e)}"
        )

    # ---------------------------------------------------------
    # 6. Calculate processing time
    # ---------------------------------------------------------
    processing_time_ms = round(
        (time.time() - start_time) * 1000,
        2
    )

    # ---------------------------------------------------------
    # 7. Audio metadata
    # ---------------------------------------------------------
    duration = round(
        float(len(y) / sr),
        3
    )

    sampling_rate = int(sr)

    # ---------------------------------------------------------
    # 8. Build final API response
    # ---------------------------------------------------------
    response_payload = {
        "success": True,

        # Prediction
        "prediction": prediction_result.get(
            "prediction"
        ),

        "is_human": prediction_result.get(
            "is_human"
        ),

        "confidence": prediction_result.get(
            "confidence"
        ),

        # Model information
        "demo_mode": prediction_result.get(
            "demo_mode",
            False
        ),

        "model_used": prediction_result.get(
            "model_used"
        ),

        "status_note": prediction_result.get(
            "status_note"
        ),

        # Audio analysis
        "audio_metrics": prediction_result.get(
            "metrics",
            {}
        ),

        "feature_summary": prediction_result.get(
            "feature_summary",
            {}
        ),

        # Explainable AI
        "xai_explanation": prediction_result.get(
            "xai_explanation",
            {}
        ),

        # Visualization
        "waveform_data": waveform_points,

        # Audio metadata
        "duration": duration,

        "sampling_rate": sampling_rate,

        # Performance
        "processing_time_ms": processing_time_ms
    }

    return response_payload


def generate_waveform_points(y, num_points=100):
    """
    Generates downsampled amplitude values for the
    frontend waveform visualization.
    """

    if len(y) == 0:
        return [0.1] * num_points

    step = max(
        1,
        len(y) // num_points
    )

    waveform = []

    for i in range(num_points):

        start = i * step
        end = (i + 1) * step

        chunk = y[start:end]

        if len(chunk) > 0:
            value = float(
                np.max(
                    np.abs(chunk)
                )
            )
        else:
            value = 0.0

        # Keep waveform values within 0-1
        value = min(
            1.0,
            max(0.0, value)
        )

        waveform.append(
            round(value, 3)
        )

    return waveform