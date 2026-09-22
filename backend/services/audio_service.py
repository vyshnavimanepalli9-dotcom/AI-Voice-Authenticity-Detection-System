import time
import librosa
import numpy as np
import logging
from ml.preprocessing import validate_audio_signal, normalize_signal
from ml.predict import predict_voice_authenticity

logger = logging.getLogger(__name__)

def process_and_analyze_audio(file_path):
    """
    Loads audio file, performs signal validation, resampling, feature extraction,
    runs authenticity classifier, and generates visual waveform data points.
    """
    start_time = time.time()

    # 1. Load Audio File using Librosa (default sampling rate 22050 Hz, mono)
    target_sr = 22050
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as e:
        logger.error(f"Librosa audio loading error: {e}")
        raise ValueError(f"Failed to decode audio file. Ensure format is supported. Error: {str(e)}")

    # 2. Signal Validation & Quality Check
    is_valid, warning_msg = validate_audio_signal(y, sr)
    if not is_valid:
        raise ValueError(warning_msg)

    # 3. Audio Signal Normalization
    y = normalize_signal(y)

    # 4. Generate Downsampled Waveform Points for UI Visualizer (100 points)
    waveform_points = generate_waveform_points(y, num_points=100)

    # 5. Run ML Authenticity Classifier
    prediction_result = predict_voice_authenticity(y, sr)

    # 6. Compute total processing time
    processing_time_ms = round((time.time() - start_time) * 1000, 2)

    # Package Final Payload
    rresponse_payload = {
    "success": True,
    "prediction": prediction_result["prediction"],
    "is_human": prediction_result["is_human"],
    "confidence": prediction_result["confidence"],
    "demo_mode": prediction_result["demo_mode"],
    "model_used": prediction_result["model_used"],
    "status_note": prediction_result["status_note"],
    "audio_metrics": prediction_result["metrics"],
    "feature_summary": prediction_result["feature_summary"],
    "xai_explanation": prediction_result["xai_explanation"],
    "waveform_data": waveform_points,
    "processing_time_ms": processing_time_ms,

    # Database-friendly metadata
    "duration": round(float(len(y) / sr), 3),
    "sampling_rate": sr
    
     "processing_time_ms": processing_time_ms
}

    return response_payload

def generate_waveform_points(y, num_points=100):
    """Generates downsampled amplitude values [0.0 - 1.0] for UI audio waveform chart rendering."""
    if len(y) == 0:
        return [0.1] * num_points
    
    step = max(1, len(y) // num_points)
    waveform = []
    for i in range(num_points):
        chunk = y[i * step : (i + 1) * step]
        if len(chunk) > 0:
            val = float(np.max(np.abs(chunk)))
        else:
            val = 0.0
        waveform.append(round(val, 3))
    return waveform
