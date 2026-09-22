import os
import logging
import math

import librosa
import numpy as np
import soundfile as sf

from scipy.signal import resample_poly


logger = logging.getLogger(__name__)


# =========================================================
# Configuration
# =========================================================

TARGET_SAMPLE_RATE = 22050

# Maximum audio duration processed by the feature extractor.
MAX_AUDIO_DURATION_SECONDS = 60.0

# Standard feature extraction parameters
FRAME_LENGTH = 2048
HOP_LENGTH = 512


# =========================================================
# Utility: Safe Audio Normalization
# =========================================================

def _normalize_audio(y):
    """
    Clean and normalize an audio signal.

    Returns:
        numpy.ndarray
    """

    y = np.asarray(y, dtype=np.float32)

    if y.size == 0:
        return y

    y = np.nan_to_num(
        y,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    max_amplitude = float(
        np.max(np.abs(y))
    )

    if max_amplitude > 1.0:
        y = (
            y / max_amplitude
        ).astype(np.float32)

    return y


# =========================================================
# Utility: Frame-based Zero Crossing Rate
# =========================================================

def _calculate_zero_crossing_rate(
    y,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
):
    """
    NumPy implementation of frame-based Zero Crossing Rate.

    We intentionally do not use:
        librosa.feature.zero_crossing_rate()

    This keeps ZCR independent from librosa's Numba-based
    implementation.

    Returns:
        1D numpy array
    """

    y = np.asarray(
        y,
        dtype=np.float32
    )

    if len(y) == 0:
        return np.array(
            [0.0],
            dtype=np.float32
        )

    if len(y) < frame_length:
        y = np.pad(
            y,
            (
                0,
                frame_length - len(y)
            ),
            mode="constant"
        )

    frames = np.lib.stride_tricks.sliding_window_view(
        y,
        frame_length
    )

    frames = frames[::hop_length]

    if frames.size == 0:
        return np.array(
            [0.0],
            dtype=np.float32
        )

    signs = np.signbit(frames)

    crossings = (
        signs[:, :-1] !=
        signs[:, 1:]
    )

    zcr = np.mean(
        crossings,
        axis=1
    ).astype(np.float32)

    return zcr


# =========================================================
# Utility: Safe Float
# =========================================================

def _safe_float(
    value,
    default=0.0
):
    """
    Convert a numerical value into a safe Python float.
    """

    try:
        value = float(value)

        if not np.isfinite(value):
            return float(default)

        return value

    except Exception:
        return float(default)


# =========================================================
# Main Feature Extraction
# =========================================================

def extract_features_from_audio(
    y,
    sr
):
    """
    Extract acoustic features from an audio signal.

    Feature vector contains exactly 79 features:

        1-20    MFCC mean
        21-40   MFCC standard deviation
        41      Mel mean
        42      Mel standard deviation
        43      Spectral centroid mean
        44      Spectral centroid standard deviation
        45      Spectral bandwidth mean
        46      Spectral bandwidth standard deviation
        47      Spectral rolloff mean
        48      Spectral rolloff standard deviation
        49      ZCR mean
        50      ZCR standard deviation
        51-62   Chroma mean
        63-74   Chroma standard deviation
        75      F0 mean
        76      F0 standard deviation
        77      F0 variation
        78      MFCC delta standard deviation
        79      Spectral flux

    Returns:
        {
            "feature_vector": numpy array,
            "feature_summary": dictionary,
            "signal_metrics": dictionary
        }
    """

    # =====================================================
    # 0. Validate input
    # =====================================================

    if y is None:
        raise ValueError(
            "Audio signal is None."
        )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    if len(y) == 0:
        raise ValueError(
            "Audio signal is empty."
        )

    if sr is None or sr <= 0:
        raise ValueError(
            "Invalid sample rate."
        )

    sr = int(sr)

    # =====================================================
    # Clean audio
    # =====================================================

    y = _normalize_audio(y)

    if len(y) == 0:
        raise ValueError(
            "Audio signal contains no usable samples."
        )

    # =====================================================
    # Limit extremely long audio
    # =====================================================

    max_samples = int(
        MAX_AUDIO_DURATION_SECONDS * sr
    )

    if len(y) > max_samples:

        logger.warning(
            "Audio duration exceeds %.1f seconds. "
            "Only the first %.1f seconds will be analyzed.",
            MAX_AUDIO_DURATION_SECONDS,
            MAX_AUDIO_DURATION_SECONDS
        )

        y = y[:max_samples]

    # =====================================================
    # 1. Signal Metrics
    # =====================================================

    duration = (
        len(y) /
        float(sr)
    )

    rms = _safe_float(
        np.sqrt(
            np.mean(
                np.square(y)
            )
        )
    )

    # =====================================================
    # 2. Silence Detection
    # =====================================================

    try:

        intervals = librosa.effects.split(
            y,
            top_db=25
        )

        if len(intervals) > 0:

            non_silent_samples = int(
                sum(
                    end - start
                    for start, end in intervals
                )
            )

        else:

            non_silent_samples = 0

    except Exception as e:

        logger.warning(
            "Silence detection fallback: %s",
            e
        )

        threshold = max(
            float(
                np.max(
                    np.abs(y)
                )
            ) * 0.02,
            1e-5
        )

        non_silent_samples = int(
            np.sum(
                np.abs(y) > threshold
            )
        )

    silence_ratio = float(
        max(
            0.0,
            min(
                1.0,
                1.0 -
                (
                    non_silent_samples /
                    max(len(y), 1)
                )
            )
        )
    )

    # =====================================================
    # 3. MFCC Features
    # =====================================================

    try:

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        mfcc_mean = np.mean(
            mfcc,
            axis=1
        )

        mfcc_std = np.std(
            mfcc,
            axis=1
        )

    except Exception as e:

        logger.error(
            "MFCC extraction failed: %s",
            e,
            exc_info=True
        )

        raise RuntimeError(
            "MFCC feature extraction failed."
        ) from e

    # =====================================================
    # 4. MFCC Temporal Dynamics
    # =====================================================

    try:

        if mfcc.shape[1] >= 3:

            mfcc_delta = librosa.feature.delta(
                mfcc
            )

            delta_std = _safe_float(
                np.mean(
                    np.std(
                        mfcc_delta,
                        axis=1
                    )
                )
            )

        else:

            delta_std = 0.0

    except Exception as e:

        logger.warning(
            "MFCC delta extraction fallback: %s",
            e
        )

        delta_std = 0.0

    # =====================================================
    # 5. Mel Spectrogram
    # =====================================================

    try:

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=64,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        mel_spec_db = librosa.power_to_db(
            mel_spec,
            ref=np.max
        )

        mel_mean = _safe_float(
            np.mean(mel_spec_db)
        )

        mel_std = _safe_float(
            np.std(mel_spec_db)
        )

    except Exception as e:

        logger.error(
            "Mel spectrogram extraction failed: %s",
            e,
            exc_info=True
        )

        raise RuntimeError(
            "Mel spectrogram extraction failed."
        ) from e

    # =====================================================
    # 6. Spectral Centroid
    # =====================================================

    try:

        spec_cent = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )[0]

        cent_mean = _safe_float(
            np.mean(spec_cent)
        )

        cent_std = _safe_float(
            np.std(spec_cent)
        )

    except Exception as e:

        logger.warning(
            "Spectral centroid fallback: %s",
            e
        )

        cent_mean = 0.0
        cent_std = 0.0

    # =====================================================
    # 7. Spectral Bandwidth
    # =====================================================

    try:

        spec_bw = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )[0]

        bw_mean = _safe_float(
            np.mean(spec_bw)
        )

        bw_std = _safe_float(
            np.std(spec_bw)
        )

    except Exception as e:

        logger.warning(
            "Spectral bandwidth fallback: %s",
            e
        )

        bw_mean = 0.0
        bw_std = 0.0

    # =====================================================
    # 8. Spectral Rolloff
    # =====================================================

    try:

        spec_roll = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            roll_percent=0.85,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )[0]

        roll_mean = _safe_float(
            np.mean(spec_roll)
        )

        roll_std = _safe_float(
            np.std(spec_roll)
        )

    except Exception as e:

        logger.warning(
            "Spectral rolloff fallback: %s",
            e
        )

        roll_mean = 0.0
        roll_std = 0.0

    # =====================================================
    # 9. Spectral Flux
    # =====================================================

    try:

        stft_complex = librosa.stft(
            y,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        stft = np.abs(
            stft_complex
        )

        if stft.shape[1] > 1:

            frame_difference = np.diff(
                stft,
                axis=1
            )

            spectral_flux = _safe_float(
                np.mean(
                    np.sqrt(
                        np.sum(
                            frame_difference ** 2,
                            axis=0
                        )
                    )
                )
            )

        else:

            spectral_flux = 0.0

    except Exception as e:

        logger.warning(
            "Spectral flux fallback: %s",
            e
        )

        spectral_flux = 0.0

    # =====================================================
    # 10. Zero Crossing Rate
    # =====================================================

    try:

        zcr = _calculate_zero_crossing_rate(
            y,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        zcr_mean = _safe_float(
            np.mean(zcr)
        )

        zcr_std = _safe_float(
            np.std(zcr)
        )

    except Exception as e:

        logger.warning(
            "Zero crossing rate fallback: %s",
            e
        )

        zcr_mean = 0.0
        zcr_std = 0.0

    # =====================================================
    # 11. Chroma Features
    # =====================================================

    try:

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            tuning=0.0,
            n_chroma=12,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        chroma_mean = np.mean(
            chroma,
            axis=1
        )

        chroma_std = np.std(
            chroma,
            axis=1
        )

    except Exception as e:

        logger.warning(
            "Chroma extraction fallback: %s",
            e
        )

        chroma_mean = np.zeros(
            12,
            dtype=np.float32
        )

        chroma_std = np.zeros(
            12,
            dtype=np.float32
        )

    # =====================================================
    # 12. Pitch / Fundamental Frequency
    # =====================================================

    try:

        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sr,
            fmin=50,
            fmax=500,
            n_fft=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )

        pitch_values = []

        for t in range(
            pitches.shape[1]
        ):

            magnitude_column = magnitudes[:, t]

            if len(magnitude_column) == 0:
                continue

            index = int(
                np.argmax(
                    magnitude_column
                )
            )

            pitch = _safe_float(
                pitches[index, t]
            )

            magnitude = _safe_float(
                magnitudes[index, t]
            )

            if (
                pitch >= 50.0
                and
                pitch <= 500.0
                and
                magnitude > 0.0
            ):

                pitch_values.append(
                    pitch
                )

        if len(pitch_values) > 0:

            pitch_values = np.asarray(
                pitch_values,
                dtype=np.float32
            )

            f0_mean = _safe_float(
                np.mean(
                    pitch_values
                ),
                default=140.0
            )

            f0_std = _safe_float(
                np.std(
                    pitch_values
                ),
                default=12.0
            )

            f0_var = _safe_float(
                f0_std /
                (
                    f0_mean +
                    1e-6
                ),
                default=0.08
            )

        else:

            f0_mean = 140.0
            f0_std = 12.0
            f0_var = 0.08

    except Exception as e:

        logger.warning(
            "Pitch extraction fallback: %s",
            e
        )

        f0_mean = 140.0
        f0_std = 12.0
        f0_var = 0.08

    # =====================================================
    # 13. Clean Numerical Values
    # =====================================================

    mfcc_mean = np.nan_to_num(
        mfcc_mean,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    mfcc_std = np.nan_to_num(
        mfcc_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    chroma_mean = np.nan_to_num(
        chroma_mean,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    chroma_std = np.nan_to_num(
        chroma_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    # =====================================================
    # 14. ML Feature Vector
    # =====================================================

    feature_vector = np.hstack([

        mfcc_mean,
        mfcc_std,

        mel_mean,
        mel_std,

        cent_mean,
        cent_std,

        bw_mean,
        bw_std,

        roll_mean,
        roll_std,

        zcr_mean,
        zcr_std,

        chroma_mean,
        chroma_std,

        f0_mean,
        f0_std,
        f0_var,

        delta_std,

        spectral_flux
    ])

    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32
    )

    feature_vector = np.nan_to_num(
        feature_vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # =====================================================
    # 15. Feature Count Validation
    # =====================================================

    expected_feature_count = 79

    if len(feature_vector) != expected_feature_count:

        raise RuntimeError(
            f"Feature extraction produced "
            f"{len(feature_vector)} features, "
            f"but the trained model expects "
            f"{expected_feature_count}."
        )

    # =====================================================
    # 16. Human-readable Feature Summary
    # =====================================================

    feature_summary = {

        "mfcc_mean_1": _safe_float(
            mfcc_mean[0]
        ),

        "mfcc_variance": _safe_float(
            np.mean(
                mfcc_std
            )
        ),

        "delta_mfcc_std": round(
            delta_std,
            4
        ),

        "spectral_flux": round(
            spectral_flux,
            4
        ),

        "mel_energy_db": round(
            mel_mean,
            2
        ),

        "spectral_centroid_hz": round(
            cent_mean,
            2
        ),

        "spectral_bandwidth_hz": round(
            bw_mean,
            2
        ),

        "spectral_rolloff_hz": round(
            roll_mean,
            2
        ),

        "zero_crossing_rate": round(
            zcr_mean,
            4
        ),

        "chroma_energy": round(
            _safe_float(
                np.mean(
                    chroma_mean
                )
            ),
            4
        ),

        "pitch_f0_mean_hz": round(
            f0_mean,
            2
        ),

        "pitch_std_hz": round(
            f0_std,
            2
        ),

        "pitch_variation_index": round(
            f0_var,
            4
        ),

        "silence_ratio": round(
            silence_ratio,
            4
        ),

        "rms_energy": round(
            rms,
            4
        )
    }

    # =====================================================
    # 17. Signal Metrics
    # =====================================================

    signal_metrics = {

        "duration_seconds": round(
            duration,
            2
        ),

        "sample_rate_hz": int(
            sr
        ),

        "total_samples": int(
            len(y)
        ),

        "silence_ratio": round(
            silence_ratio,
            3
        ),

        "feature_count": int(
            len(feature_vector)
        )
    }

    # =====================================================
    # 18. Final Result
    # =====================================================

    return {

        "feature_vector": feature_vector,

        "feature_summary": feature_summary,

        "signal_metrics": signal_metrics
    }


# =========================================================
# Extract Features Directly From File
# =========================================================

def extract_from_file(
    file_path,
    target_sr=TARGET_SAMPLE_RATE
):
    """
    Load an audio file and extract its acoustic features.

    Uses:

        SoundFile
            ↓
        NumPy
            ↓
        SciPy resample_poly
            ↓
        22050 Hz mono audio
            ↓
        Feature extraction
    """

    try:

        # =================================================
        # 1. Validate file
        # =================================================

        if not file_path:

            raise ValueError(
                "Audio file path is empty."
            )

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Audio file not found: {file_path}"
            )

        # =================================================
        # 2. Load audio with SoundFile
        # =================================================

        y, original_sr = sf.read(
            file_path,
            dtype="float32",
            always_2d=False
        )

        # =================================================
        # 3. Validate loaded audio
        # =================================================

        if y is None:

            raise ValueError(
                "SoundFile returned no audio data."
            )

        y = np.asarray(
            y,
            dtype=np.float32
        )

        if y.size == 0:

            raise ValueError(
                "Audio file contains no samples."
            )

        if original_sr is None or original_sr <= 0:

            raise ValueError(
                f"Invalid source sample rate: "
                f"{original_sr}"
            )

        # =================================================
        # 4. Convert stereo/multichannel → mono
        # =================================================

        if y.ndim == 2:

            y = np.mean(
                y,
                axis=1
            ).astype(
                np.float32
            )

        elif y.ndim > 2:

            raise ValueError(
                "Unsupported audio format with more than "
                "2 dimensions."
            )

        # =================================================
        # 5. Clean numerical values
        # =================================================

        y = _normalize_audio(y)

        # =================================================
        # 6. Convert sample rate
        # =================================================

        original_sr = int(
            original_sr
        )

        target_sr = int(
            target_sr
        )

        if target_sr <= 0:

            raise ValueError(
                f"Invalid target sample rate: "
                f"{target_sr}"
            )

        if original_sr != target_sr:

            logger.info(
                "Resampling audio: %s Hz -> %s Hz",
                original_sr,
                target_sr
            )

            gcd = math.gcd(
                original_sr,
                target_sr
            )

            up = (
                target_sr //
                gcd
            )

            down = (
                original_sr //
                gcd
            )

            y = resample_poly(
                y,
                up,
                down
            ).astype(
                np.float32
            )

            sr = target_sr

        else:

            sr = original_sr

        # =================================================
        # 7. Final validation
        # =================================================

        if len(y) == 0:

            raise ValueError(
                "Audio became empty after resampling."
            )

        y = np.asarray(
            y,
            dtype=np.float32
        ).reshape(-1)

        y = _normalize_audio(y)

        # =================================================
        # 8. Limit audio duration
        # =================================================

        max_samples = int(
            MAX_AUDIO_DURATION_SECONDS * sr
        )

        if len(y) > max_samples:

            logger.warning(
                "Input audio is %.2f seconds. "
                "Limiting analysis to %.2f seconds.",
                len(y) / float(sr),
                MAX_AUDIO_DURATION_SECONDS
            )

            y = y[:max_samples]

        # =================================================
        # 9. Log successful loading
        # =================================================

        duration = (
            len(y) /
            float(sr)
        )

        logger.info(
            "Audio loaded successfully: "
            "samples=%d, sample_rate=%d, duration=%.2fs",
            len(y),
            sr,
            duration
        )

        # =================================================
        # 10. Extract acoustic features
        # =================================================

        result = extract_features_from_audio(
            y,
            sr
        )

        logger.info(
            "Feature extraction completed successfully: "
            "%d features",
            len(
                result[
                    "feature_vector"
                ]
            )
        )

        return result

    except Exception as e:

        logger.error(
            "Could not process audio file %s: %s",
            file_path,
            e,
            exc_info=True
        )

        raise RuntimeError(
            f"Unable to load audio file: {e}"
        ) from e