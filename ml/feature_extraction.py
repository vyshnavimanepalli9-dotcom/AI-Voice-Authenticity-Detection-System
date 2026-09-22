import os

# Prevent Numba compilation problems inside librosa.core.audio.
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numba

# librosa.feature imports zero-crossing helpers that use Numba
# decorators during module initialization. Our project already
# calculates ZCR with a NumPy implementation, so these decorators
# do not need to compile at runtime.
def _no_compile_decorator(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

numba.guvectorize = _no_compile_decorator
numba.stencil = _no_compile_decorator

import logging

import librosa
import numpy as np
import soundfile as sf

from scipy.signal import resample_poly

logger = logging.getLogger(__name__)# =========================================================
# Utility: Frame-based Zero Crossing Rate
# =========================================================

def _calculate_zero_crossing_rate(
    y,
    frame_length=2048,
    hop_length=512
):
    """
    NumPy implementation of frame-based Zero Crossing Rate.

    This intentionally avoids:
        librosa.feature.zero_crossing_rate()

    because the Render deployment was producing:

        no compiled object yet for <Library '_zc_wrapper'>

    Returns:
        zcr -> 1D numpy array
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

    # -----------------------------------------------------
    # Pad short audio
    # -----------------------------------------------------

    if len(y) < frame_length:

        y = np.pad(
            y,
            (
                0,
                frame_length - len(y)
            ),
            mode="constant"
        )

    # -----------------------------------------------------
    # Number of frames
    # -----------------------------------------------------

    number_of_frames = (
        1
        +
        (len(y) - frame_length)
        // hop_length
    )

    # -----------------------------------------------------
    # Create overlapping frames
    # -----------------------------------------------------

    frames = np.lib.stride_tricks.sliding_window_view(
        y,
        frame_length
    )

    frames = frames[
        ::hop_length
    ]

    frames = frames[
        :number_of_frames
    ]

    # -----------------------------------------------------
    # Detect sign changes
    # -----------------------------------------------------

    signs = np.signbit(
        frames
    )

    crossings = (
        signs[:, :-1]
        !=
        signs[:, 1:]
    )

    # -----------------------------------------------------
    # Calculate crossing ratio
    # -----------------------------------------------------

    zcr = (
        np.mean(
            crossings,
            axis=1
        )
        .astype(np.float32)
    )

    return zcr


# =========================================================
# Main Feature Extraction
# =========================================================

def extract_features_from_audio(
    y,
    sr
):
    """
    Extract acoustic features from an audio signal.

    Feature vector contains:

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

    Total:
        79 features

    Returns:
        {
            "feature_vector": numpy array,
            "feature_summary": dictionary,
            "signal_metrics": dictionary
        }
    """

    # =====================================================
    # 0. Validate Input
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

    # Remove NaN / infinity
    y = np.nan_to_num(
        y,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Normalize extremely large values
    max_amplitude = np.max(
        np.abs(y)
    )

    if max_amplitude > 1.0:

        y = (
            y /
            max_amplitude
        )

    # =====================================================
    # 1. Signal Metrics
    # =====================================================

    duration = float(
        len(y) / sr
    )

    # -----------------------------------------------------
    # RMS Energy
    # -----------------------------------------------------

    rms = float(
        np.sqrt(
            np.mean(
                np.square(y)
            )
        )
    )

    # -----------------------------------------------------
    # Silence Detection
    #
    # Keep librosa.effects.split because it uses an
    # energy-based silence calculation and is separate
    # from the problematic ZCR function.
    # -----------------------------------------------------

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

        # Simple RMS-based fallback
        threshold = (
            max(
                np.max(
                    np.abs(y)
                ) * 0.02,
                1e-5
            )
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
    # 2. MFCC Features
    # =====================================================

    try:

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=20
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
            e
        )

        raise RuntimeError(
            "MFCC feature extraction failed."
        ) from e

    # =====================================================
    # 3. MFCC Temporal Dynamics
    # =====================================================

    try:

        # librosa.delta requires enough frames.
        if mfcc.shape[1] >= 3:

            mfcc_delta = librosa.feature.delta(
                mfcc
            )

            delta_std = float(
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
    # 4. Mel Spectrogram
    # =====================================================

    try:

        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=64
        )

        mel_spec_db = librosa.power_to_db(
            mel_spec,
            ref=np.max
        )

        mel_mean = float(
            np.mean(
                mel_spec_db
            )
        )

        mel_std = float(
            np.std(
                mel_spec_db
            )
        )

    except Exception as e:

        logger.error(
            "Mel spectrogram extraction failed: %s",
            e
        )

        raise RuntimeError(
            "Mel spectrogram extraction failed."
        ) from e

    # =====================================================
    # 5. Spectral Centroid
    # =====================================================

    try:

        spec_cent = librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )[0]

        cent_mean = float(
            np.mean(
                spec_cent
            )
        )

        cent_std = float(
            np.std(
                spec_cent
            )
        )

    except Exception as e:

        logger.warning(
            "Spectral centroid fallback: %s",
            e
        )

        cent_mean = 0.0
        cent_std = 0.0

    # =====================================================
    # 6. Spectral Bandwidth
    # =====================================================

    try:

        spec_bw = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )[0]

        bw_mean = float(
            np.mean(
                spec_bw
            )
        )

        bw_std = float(
            np.std(
                spec_bw
            )
        )

    except Exception as e:

        logger.warning(
            "Spectral bandwidth fallback: %s",
            e
        )

        bw_mean = 0.0
        bw_std = 0.0

    # =====================================================
    # 7. Spectral Rolloff
    # =====================================================

    try:

        spec_roll = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            roll_percent=0.85
        )[0]

        roll_mean = float(
            np.mean(
                spec_roll
            )
        )

        roll_std = float(
            np.std(
                spec_roll
            )
        )

    except Exception as e:

        logger.warning(
            "Spectral rolloff fallback: %s",
            e
        )

        roll_mean = 0.0
        roll_std = 0.0

    # =====================================================
    # 8. Spectral Flux
    # =====================================================

    try:

        stft_complex = librosa.stft(
            y
        )

        stft = np.abs(
            stft_complex
        )

        if stft.shape[1] > 1:

            frame_difference = np.diff(
                stft,
                axis=1
            )

            spectral_flux = float(
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
    # 9. Zero Crossing Rate
    # =====================================================
    #
    # IMPORTANT:
    #
    # We intentionally DO NOT use:
    #
    # librosa.feature.zero_crossing_rate()
    #
    # because Render produced:
    #
    # no compiled object yet for
    # <Library '_zc_wrapper' ...>
    #
    # Instead we use our NumPy implementation above.
    #
    # =====================================================

    try:

        zcr = _calculate_zero_crossing_rate(
            y,
            frame_length=2048,
            hop_length=512
        )

        zcr_mean = float(
            np.mean(
                zcr
            )
        )

        zcr_std = float(
            np.std(
                zcr
            )
        )

    except Exception as e:

        logger.warning(
            "Zero crossing rate fallback: %s",
            e
        )

        zcr_mean = 0.0
        zcr_std = 0.0

    # =====================================================
    # 10. Chroma Features
    # =====================================================

    try:

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            tuning=0.0,
            n_chroma=12
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
    # 11. Pitch / Fundamental Frequency
    # =====================================================

    try:

        pitches, magnitudes = librosa.piptrack(
            y=y,
            sr=sr,
            fmin=50,
            fmax=500
        )

        pitch_values = []

        for t in range(
            pitches.shape[1]
        ):

            magnitude_column = magnitudes[
                :,
                t
            ]

            if len(magnitude_column) == 0:
                continue

            index = int(
                np.argmax(
                    magnitude_column
                )
            )

            pitch = float(
                pitches[
                    index,
                    t
                ]
            )

            magnitude = float(
                magnitudes[
                    index,
                    t
                ]
            )

            # Accept only meaningful pitch estimates
            if (
                pitch >= 50
                and
                pitch <= 500
                and
                magnitude > 0
            ):

                pitch_values.append(
                    pitch
                )

        # -------------------------------------------------
        # Calculate pitch statistics
        # -------------------------------------------------

        if len(pitch_values) > 0:

            pitch_values = np.asarray(
                pitch_values,
                dtype=np.float32
            )

            f0_mean = float(
                np.mean(
                    pitch_values
                )
            )

            f0_std = float(
                np.std(
                    pitch_values
                )
            )

            f0_var = float(
                f0_std /
                (
                    f0_mean +
                    1e-6
                )
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
    # 12. Clean Numerical Values
    # =====================================================

    mfcc_mean = np.nan_to_num(
        mfcc_mean,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mfcc_std = np.nan_to_num(
        mfcc_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    chroma_mean = np.nan_to_num(
        chroma_mean,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    chroma_std = np.nan_to_num(
        chroma_std,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # =====================================================
    # 13. ML Feature Vector
    # =====================================================
    #
    # IMPORTANT:
    #
    # The order is preserved from your existing pipeline.
    #
    # Total = 79 features.
    #
    # =====================================================

    feature_vector = np.hstack([

        # MFCC
        mfcc_mean,
        mfcc_std,

        # Mel
        mel_mean,
        mel_std,

        # Spectral centroid
        cent_mean,
        cent_std,

        # Spectral bandwidth
        bw_mean,
        bw_std,

        # Spectral rolloff
        roll_mean,
        roll_std,

        # Zero crossing rate
        zcr_mean,
        zcr_std,

        # Chroma
        chroma_mean,
        chroma_std,

        # Pitch
        f0_mean,
        f0_std,
        f0_var,

        # MFCC dynamics
        delta_std,

        # Spectral flux
        spectral_flux
    ])

    # Ensure float32
    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32
    )

    # Ensure no NaN / infinity
    feature_vector = np.nan_to_num(
        feature_vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # =====================================================
    # 14. Feature Count Validation
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
    # 15. Human-readable Feature Summary
    # =====================================================

    feature_summary = {

        "mfcc_mean_1": float(
            mfcc_mean[0]
        ),

        "mfcc_variance": float(
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
            float(
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
    # 16. Signal Metrics
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
        )
    }

    # =====================================================
    # 17. Final Result
    # =====================================================

    return {

        "feature_vector": feature_vector,

        "feature_summary": feature_summary,

        "signal_metrics": signal_metrics
    }


# =========================================================
# Extract Features Directly From File
# =========================================================

# =========================================================
# Extract Features Directly From File
# =========================================================

def extract_from_file(
    file_path,
    target_sr=22050
):
    """
    Load an audio file and extract its
    acoustic feature dictionary.

    IMPORTANT:
        This function intentionally does NOT use
        librosa.load().

    librosa.load() was triggering a Numba compilation
    error inside librosa.core.audio on the deployment
    environment.

    Instead:

        SoundFile
            ↓
        NumPy
            ↓
        SciPy resample_poly
            ↓
        22050 Hz mono audio
            ↓
        feature extraction

    Parameters:
        file_path:
            Path to audio file.

        target_sr:
            Target sample rate.
            Default = 22050 Hz.

    Returns:
        Feature extraction dictionary.
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
        # 2. Load audio using SoundFile
        # =================================================
        #
        # IMPORTANT:
        #
        # Do NOT use:
        #
        #     librosa.load()
        #
        # because it triggers the Numba/librosa
        # compilation problem seen in the project.
        #
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
                f"Invalid source sample rate: {original_sr}"
            )

        # =================================================
        # 4. Convert stereo/multichannel → mono
        # =================================================

        if y.ndim == 2:

            # SoundFile normally returns:
            #
            #     samples × channels
            #
            # Average all channels.

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
        # 5. Clean invalid numerical values
        # =================================================

        y = np.nan_to_num(
            y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # =================================================
        # 6. Resample if necessary
        # =================================================

        original_sr = int(
            original_sr
        )

        target_sr = int(
            target_sr
        )

        if target_sr <= 0:
            raise ValueError(
                f"Invalid target sample rate: {target_sr}"
            )

        if original_sr != target_sr:

            logger.info(
                "Resampling audio: %s Hz -> %s Hz",
                original_sr,
                target_sr
            )

            # -------------------------------------------------
            # Calculate rational resampling ratio
            # -------------------------------------------------

            import math

            gcd = math.gcd(
                original_sr,
                target_sr
            )

            up = target_sr // gcd
            down = original_sr // gcd

            # -------------------------------------------------
            # Polyphase resampling
            # -------------------------------------------------

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
        # 7. Final audio validation
        # =================================================

        if len(y) == 0:
            raise ValueError(
                "Audio became empty after resampling."
            )

        # Ensure one-dimensional signal

        y = np.asarray(
            y,
            dtype=np.float32
        ).reshape(
            -1
        )

        # Remove invalid values again after resampling

        y = np.nan_to_num(
            y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # =================================================
        # 8. Normalize extremely large values
        # =================================================

        max_amplitude = float(
            np.max(
                np.abs(y)
            )
        )

        if max_amplitude > 1.0:

            y = (
                y /
                max_amplitude
            ).astype(
                np.float32
            )

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

        return extract_features_from_audio(
            y,
            sr
        )

    except Exception as e:

        logger.error(
            "Could not load audio file %s: %s",
            file_path,
            e,
            exc_info=True
        )

        raise RuntimeError(
            f"Unable to load audio file: {e}"
        ) from e