"""
Production-safe audio feature extraction.

AI Voice Authenticity Detection System

IMPORTANT:
- No librosa
- No Numba
- No librosa.effects.split()
- No librosa.zero_crossings()
- No librosa.feature.zero_crossing_rate()
- Uses NumPy + SciPy + SoundFile
- Exactly 79 features

Feature layout:

1-13   : 13 MFCC means
14-26  : 13 MFCC standard deviations
27-39  : 13 MFCC delta means
40-66  : 27 additional acoustic features
67-71  : 5 spectral statistics
72-75  : 4 temporal statistics
76-79  : 4 RMS dynamics

13 + 13 + 13 + 27 + 5 + 4 + 4 = 79
"""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf

from scipy.fft import dct
from scipy.signal import resample_poly


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SAMPLE_RATE = 22050
MAX_AUDIO_DURATION_SECONDS = 60.0

FRAME_LENGTH = 2048
HOP_LENGTH = 512

N_MFCC = 13
N_MELS = 40

EXPECTED_FEATURE_COUNT = 79

EPSILON = 1e-10


# ============================================================
# BASIC AUDIO UTILITIES
# ============================================================

def _to_mono_float32(y):
    """
    Convert audio to a 1-D float32 mono signal.
    """

    y = np.asarray(y)

    if y.ndim == 0:
        y = y.reshape(1)

    if y.ndim == 2:

        # soundfile normally gives:
        # samples x channels

        if y.shape[0] >= y.shape[1]:
            y = np.mean(y, axis=1)
        else:
            y = np.mean(y, axis=0)

    y = np.asarray(
        y,
        dtype=np.float32
    ).reshape(-1)

    y = np.nan_to_num(
        y,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return y


def _resample_audio(y, sr, target_sr):
    """
    Resample audio using scipy only.

    No librosa.load().
    """

    sr = int(sr)

    if sr <= 0:
        raise ValueError("Invalid sample rate.")

    if sr == target_sr:
        return (
            np.asarray(y, dtype=np.float32),
            sr
        )

    divisor = gcd(sr, target_sr)

    up = target_sr // divisor
    down = sr // divisor

    y_resampled = resample_poly(
        y,
        up,
        down
    )

    return (
        np.asarray(
            y_resampled,
            dtype=np.float32
        ),
        target_sr
    )


def _normalize_signal(y):
    """
    Peak normalize audio.

    Silence is left untouched.
    """

    y = np.asarray(
        y,
        dtype=np.float32
    )

    if len(y) == 0:
        return y

    peak = float(
        np.max(
            np.abs(y)
        )
    )

    if peak <= EPSILON:
        return y

    return (
        y / peak
    ).astype(np.float32)


def _frame_signal(
    y,
    frame_length=FRAME_LENGTH,
    hop_length=HOP_LENGTH
):
    """
    Convert signal into overlapping frames.

    Output shape:

        frame_length x number_of_frames
    """

    y = np.asarray(
        y,
        dtype=np.float32
    ).reshape(-1)

    if len(y) == 0:
        return np.zeros(
            (frame_length, 1),
            dtype=np.float32
        )

    if len(y) <= frame_length:

        number_of_frames = 1

    else:

        number_of_frames = (
            int(
                np.ceil(
                    (len(y) - frame_length)
                    / hop_length
                )
            )
            + 1
        )

    total_length = (
        (number_of_frames - 1)
        * hop_length
        + frame_length
    )

    if total_length > len(y):

        y = np.pad(
            y,
            (
                0,
                total_length - len(y)
            )
        )

    starts = (
        np.arange(number_of_frames)
        * hop_length
    )

    indices = (
        starts[:, None]
        + np.arange(frame_length)[None, :]
    )

    frames = y[indices]

    return frames.T.astype(
        np.float32
    )


# ============================================================
# NUMPY-ONLY ZERO CROSSING
# ============================================================

def _calculate_zero_crossing_rate(frames):
    """
    Calculate zero-crossing rate using NumPy only.

    IMPORTANT:
    This deliberately does NOT use:

        librosa.zero_crossings()
        librosa.feature.zero_crossing_rate()
        librosa.effects.split()

    Therefore this function does not trigger
    librosa's _zc_wrapper / Numba compilation.
    """

    frames = np.asarray(
        frames,
        dtype=np.float32
    )

    if frames.ndim != 2:
        return np.zeros(
            1,
            dtype=np.float32
        )

    if frames.shape[1] < 2:

        return np.zeros(
            frames.shape[0],
            dtype=np.float32
        )

    signs = frames >= 0.0

    crossings = np.logical_xor(
        signs[:, 1:],
        signs[:, :-1]
    )

    zcr = np.mean(
        crossings,
        axis=1
    )

    return zcr.astype(
        np.float32
    )


# ============================================================
# NUMPY-ONLY SILENCE DETECTION
# ============================================================

def _detect_non_silent_audio(
    frames,
    threshold_db=-45.0
):
    """
    Detect active/silent frames using RMS.

    No librosa.
    No Numba.
    """

    frames = np.asarray(
        frames,
        dtype=np.float64
    )

    frame_rms = np.sqrt(
        np.mean(
            frames ** 2,
            axis=0
        )
        + EPSILON
    )

    rms_db = (
        20.0
        * np.log10(
            np.maximum(
                frame_rms,
                EPSILON
            )
        )
    )

    active_mask = (
        rms_db > threshold_db
    )

    if len(active_mask) == 0:

        return (
            active_mask,
            1.0
        )

    silence_ratio = (
        1.0
        - float(
            np.mean(
                active_mask
            )
        )
    )

    return (
        active_mask,
        float(
            np.clip(
                silence_ratio,
                0.0,
                1.0
            )
        )
    )


# ============================================================
# POWER SPECTROGRAM
# ============================================================

def _power_spectrogram(
    y,
    sr
):
    """
    Calculate power spectrogram using NumPy FFT.
    """

    frames = _frame_signal(
        y,
        FRAME_LENGTH,
        HOP_LENGTH
    )

    window = np.hanning(
        FRAME_LENGTH
    ).astype(
        np.float32
    )

    windowed = (
        frames
        * window[:, None]
    )

    spectrum = np.fft.rfft(
        windowed,
        n=FRAME_LENGTH,
        axis=0
    )

    magnitude = np.abs(
        spectrum
    ).astype(
        np.float32
    )

    power = (
        magnitude ** 2
    ).astype(
        np.float32
    )

    frequencies = np.fft.rfftfreq(
        FRAME_LENGTH,
        d=1.0 / sr
    )

    return (
        power,
        magnitude,
        frequencies
    )


# ============================================================
# MEL FILTER BANK
# ============================================================

def _hz_to_mel(hz):
    return (
        2595.0
        * np.log10(
            1.0
            + np.asarray(hz)
            / 700.0
        )
    )


def _mel_to_hz(mel):
    return (
        700.0
        * (
            10.0
            ** (
                np.asarray(mel)
                / 2595.0
            )
            - 1.0
        )
    )


def _mel_filter_bank(
    sr,
    n_fft,
    n_mels=N_MELS,
    fmin=20.0,
    fmax=None
):
    """
    Create triangular Mel filter banks.

    Implemented without librosa.
    """

    if fmax is None:
        fmax = sr / 2.0

    fmax = min(
        float(fmax),
        sr / 2.0
    )

    fmin = max(
        0.0,
        float(fmin)
    )

    mel_points = np.linspace(
        _hz_to_mel(fmin),
        _hz_to_mel(fmax),
        n_mels + 2
    )

    hz_points = _mel_to_hz(
        mel_points
    )

    bin_numbers = np.floor(
        (
            n_fft + 1
        )
        * hz_points
        / sr
    ).astype(int)

    number_of_bins = (
        n_fft // 2 + 1
    )

    filters = np.zeros(
        (
            n_mels,
            number_of_bins
        ),
        dtype=np.float32
    )

    for m in range(
        1,
        n_mels + 1
    ):

        left = int(
            np.clip(
                bin_numbers[m - 1],
                0,
                number_of_bins - 1
            )
        )

        center = int(
            np.clip(
                bin_numbers[m],
                0,
                number_of_bins - 1
            )
        )

        right = int(
            np.clip(
                bin_numbers[m + 1],
                0,
                number_of_bins - 1
            )
        )

        if center > left:

            filters[
                m - 1,
                left:center
            ] = np.linspace(
                0.0,
                1.0,
                center - left,
                endpoint=False,
                dtype=np.float32
            )

        if right > center:

            filters[
                m - 1,
                center:right
            ] = np.linspace(
                1.0,
                0.0,
                right - center,
                endpoint=False,
                dtype=np.float32
            )

    return filters


# ============================================================
# MFCC
# ============================================================

def _compute_mfcc(
    power_spectrogram,
    sr
):
    """
    Calculate:

        13 MFCCs
        40-band log-Mel representation

    using NumPy + SciPy only.
    """

    filters = _mel_filter_bank(
        sr=sr,
        n_fft=FRAME_LENGTH,
        n_mels=N_MELS,
        fmin=20.0,
        fmax=min(
            8000.0,
            sr / 2.0
        )
    )

    mel_energy = (
        filters
        @ power_spectrogram
    )

    log_mel = np.log(
        np.maximum(
            mel_energy,
            EPSILON
        )
    )

    mfcc_all = dct(
        log_mel,
        type=2,
        axis=0,
        norm="ortho"
    )

    mfcc = mfcc_all[
        :N_MFCC
    ].astype(
        np.float32
    )

    return (
        mfcc,
        log_mel.astype(
            np.float32
        )
    )


def _compute_delta(matrix):
    """
    Calculate first-order MFCC delta.
    """

    matrix = np.asarray(
        matrix,
        dtype=np.float32
    )

    if matrix.shape[1] < 2:

        return np.zeros_like(
            matrix
        )

    padded = np.pad(
        matrix,
        (
            (0, 0),
            (1, 1)
        ),
        mode="edge"
    )

    delta = (
        padded[:, 2:]
        - padded[:, :-2]
    ) * 0.5

    return delta.astype(
        np.float32
    )


# ============================================================
# SPECTRAL FEATURES
# ============================================================

def _spectral_centroid(
    magnitude,
    frequencies
):

    denominator = (
        np.sum(
            magnitude,
            axis=0
        )
        + EPSILON
    )

    return (
        np.sum(
            magnitude
            * frequencies[:, None],
            axis=0
        )
        / denominator
    )


def _spectral_bandwidth(
    magnitude,
    frequencies,
    centroid
):

    denominator = (
        np.sum(
            magnitude,
            axis=0
        )
        + EPSILON
    )

    variance = (
        np.sum(
            magnitude
            * (
                frequencies[:, None]
                - centroid[None, :]
            ) ** 2,
            axis=0
        )
        / denominator
    )

    return np.sqrt(
        np.maximum(
            variance,
            0.0
        )
    )


def _spectral_rolloff(
    magnitude,
    frequencies,
    percentage=0.85
):

    cumulative = np.cumsum(
        magnitude,
        axis=0
    )

    total = (
        cumulative[-1]
        + EPSILON
    )

    threshold = (
        total
        * percentage
    )

    indices = np.argmax(
        cumulative
        >= threshold[None, :],
        axis=0
    )

    indices = np.clip(
        indices,
        0,
        len(frequencies) - 1
    )

    return frequencies[
        indices
    ]


def _spectral_flux(magnitude):

    if magnitude.shape[1] < 2:
        return 0.0

    normalized = (
        magnitude
        / (
            np.sum(
                magnitude,
                axis=0,
                keepdims=True
            )
            + EPSILON
        )
    )

    differences = np.diff(
        normalized,
        axis=1
    )

    flux = np.sqrt(
        np.sum(
            differences ** 2,
            axis=0
        )
    )

    return float(
        np.mean(
            flux
        )
    )


# ============================================================
# CHROMA
# ============================================================

def _chroma_features(
    magnitude,
    frequencies
):
    """
    Calculate a simple 12-bin chroma representation.
    """

    chroma = np.zeros(
        (
            12,
            magnitude.shape[1]
        ),
        dtype=np.float32
    )

    valid = (
        frequencies >= 20.0
    )

    if not np.any(valid):
        return chroma

    valid_frequencies = (
        frequencies[valid]
    )

    valid_magnitude = (
        magnitude[valid]
    )

    midi = (
        69.0
        + 12.0
        * np.log2(
            np.maximum(
                valid_frequencies,
                20.0
            )
            / 440.0
        )
    )

    pitch_classes = np.mod(
        np.round(
            midi
        ).astype(int),
        12
    )

    for pitch_class in range(12):

        mask = (
            pitch_classes
            == pitch_class
        )

        if np.any(mask):

            chroma[
                pitch_class
            ] = np.mean(
                valid_magnitude[
                    mask
                ],
                axis=0
            )

    denominator = (
        np.sum(
            chroma,
            axis=0,
            keepdims=True
        )
        + EPSILON
    )

    chroma = (
        chroma
        / denominator
    )

    return chroma.astype(
        np.float32
    )


# ============================================================
# PITCH
# ============================================================

def _estimate_pitch(
    frames,
    sr,
    min_hz=70.0,
    max_hz=400.0
):
    """
    Lightweight autocorrelation pitch estimator.

    0 Hz means no reliable pitch was detected.
    """

    min_lag = max(
        1,
        int(
            sr / max_hz
        )
    )

    max_lag = min(
        frames.shape[0] - 1,
        int(
            sr / min_hz
        )
    )

    pitches = np.zeros(
        frames.shape[1],
        dtype=np.float32
    )

    if max_lag <= min_lag:
        return pitches

    window = np.hanning(
        frames.shape[0]
    )

    for i in range(
        frames.shape[1]
    ):

        frame = frames[
            :,
            i
        ].astype(
            np.float64
        )

        frame = (
            frame
            - np.mean(frame)
        )

        frame = (
            frame
            * window
        )

        energy = float(
            np.sum(
                frame ** 2
            )
        )

        if energy <= EPSILON:
            continue

        correlation = np.correlate(
            frame,
            frame,
            mode="full"
        )

        correlation = correlation[
            len(frame) - 1:
        ]

        search = correlation[
            min_lag:max_lag + 1
        ]

        if len(search) == 0:
            continue

        peak_offset = int(
            np.argmax(
                search
            )
        )

        lag = (
            min_lag
            + peak_offset
        )

        peak_value = float(
            search[
                peak_offset
            ]
        )

        if peak_value <= 0:
            continue

        periodicity = (
            peak_value
            / (
                float(
                    correlation[0]
                )
                + EPSILON
            )
        )

        if periodicity < 0.25:
            continue

        pitches[i] = float(
            sr / lag
        )

    return pitches


# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_features_from_audio(
    y,
    sr
):
    """
    Extract exactly 79 features.

    Public function preserved for ml.predict.py.
    """

    sr = int(sr)

    if sr <= 0:
        raise ValueError(
            "Sample rate must be greater than zero."
        )

    y = _to_mono_float32(
        y
    )

    if len(y) == 0:
        raise ValueError(
            "Audio signal is empty."
        )

    max_samples = int(
        MAX_AUDIO_DURATION_SECONDS
        * sr
    )

    if len(y) > max_samples:

        logger.info(
            "Audio exceeds %.1f seconds. "
            "Truncating.",
            MAX_AUDIO_DURATION_SECONDS
        )

        y = y[
            :max_samples
        ]

    # Resample to the same target rate
    # used by the application.
    y, sr = _resample_audio(
        y,
        sr,
        TARGET_SAMPLE_RATE
    )

    y = _normalize_signal(
        y
    )

    if len(y) == 0:
        raise ValueError(
            "Audio became empty after preprocessing."
        )

    duration_seconds = float(
        len(y) / sr
    )

    # --------------------------------------------------------
    # Frames
    # --------------------------------------------------------

    frames = _frame_signal(
        y,
        FRAME_LENGTH,
        HOP_LENGTH
    )

    # --------------------------------------------------------
    # Windowed FFT
    # --------------------------------------------------------

    window = np.hanning(
        FRAME_LENGTH
    ).astype(
        np.float32
    )

    windowed = (
        frames
        * window[:, None]
    )

    spectrum_complex = np.fft.rfft(
        windowed,
        n=FRAME_LENGTH,
        axis=0
    )

    magnitude = np.abs(
        spectrum_complex
    ).astype(
        np.float32
    )

    power = (
        magnitude ** 2
    ).astype(
        np.float32
    )

    frequencies = np.fft.rfftfreq(
        FRAME_LENGTH,
        d=1.0 / sr
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc, log_mel = _compute_mfcc(
        power,
        sr
    )

    delta_mfcc = _compute_delta(
        mfcc
    )

    # --------------------------------------------------------
    # ZCR
    # --------------------------------------------------------

    zcr = _calculate_zero_crossing_rate(
        frames
    )

    # --------------------------------------------------------
    # Silence
    # --------------------------------------------------------

    active_mask, silence_ratio = (
        _detect_non_silent_audio(
            frames,
            threshold_db=-45.0
        )
    )

    active_ratio = float(
        1.0 - silence_ratio
    )

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms_frames = np.sqrt(
        np.mean(
            frames.astype(
                np.float64
            ) ** 2,
            axis=0
        )
        + EPSILON
    ).astype(
        np.float32
    )

    rms_mean = float(
        np.mean(
            rms_frames
        )
    )

    rms_min = float(
        np.min(
            rms_frames
        )
    )

    rms_max = float(
        np.max(
            rms_frames
        )
    )

    rms_variation = float(
        np.std(
            rms_frames
        )
    )

    # --------------------------------------------------------
    # Spectral features
    # --------------------------------------------------------

    centroid = _spectral_centroid(
        magnitude,
        frequencies
    )

    bandwidth = _spectral_bandwidth(
        magnitude,
        frequencies,
        centroid
    )

    rolloff = _spectral_rolloff(
        magnitude,
        frequencies
    )

    spectral_flux = (
        _spectral_flux(
            magnitude
        )
    )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    chroma = _chroma_features(
        magnitude,
        frequencies
    )

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    pitch = _estimate_pitch(
        frames,
        sr
    )

    voiced_pitch = pitch[
        pitch > 0.0
    ]

    if len(voiced_pitch) > 0:

        pitch_mean = float(
            np.mean(
                voiced_pitch
            )
        )

        pitch_std = float(
            np.std(
                voiced_pitch
            )
        )

        pitch_min = float(
            np.min(
                voiced_pitch
            )
        )

        pitch_max = float(
            np.max(
                voiced_pitch
            )
        )

        pitch_variation_index = float(
            pitch_std
            / (
                pitch_mean
                + EPSILON
            )
        )

    else:

        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_min = 0.0
        pitch_max = 0.0
        pitch_variation_index = 0.0

    # --------------------------------------------------------
    # Signal measurements
    # --------------------------------------------------------

    peak_amplitude = float(
        np.max(
            np.abs(y)
        )
    )

    mean_amplitude = float(
        np.mean(
            np.abs(y)
        )
    )

    signal_std = float(
        np.std(
            y
        )
    )

    stft_mean = float(
        np.mean(
            magnitude
        )
    )

    stft_std = float(
        np.std(
            magnitude
        )
    )

    # ========================================================
    # FEATURES 1-13
    # 13 MFCC MEANS
    # ========================================================

    mfcc_means = np.mean(
        mfcc,
        axis=1
    )

    feature_vector = []

    feature_vector.extend(
        [
            float(value)
            for value in mfcc_means
        ]
    )

    # ========================================================
    # FEATURES 14-26
    # 13 MFCC STANDARD DEVIATIONS
    # ========================================================

    mfcc_stds = np.std(
        mfcc,
        axis=1
    )

    feature_vector.extend(
        [
            float(value)
            for value in mfcc_stds
        ]
    )

    # ========================================================
    # FEATURES 27-39
    # 13 MFCC DELTA MEANS
    # ========================================================

    delta_means = np.mean(
        delta_mfcc,
        axis=1
    )

    feature_vector.extend(
        [
            float(value)
            for value in delta_means
        ]
    )

    # ========================================================
    # FEATURES 40-66
    # 27 ADDITIONAL ACOUSTIC FEATURES
    # ========================================================

    additional_features = [

        # 40
        float(
            np.mean(
                log_mel
            )
        ),

        # 41
        float(
            np.std(
                log_mel
            )
        ),

        # 42
        float(
            np.mean(
                centroid
            )
        ),

        # 43
        float(
            np.std(
                centroid
            )
        ),

        # 44
        float(
            np.mean(
                bandwidth
            )
        ),

        # 45
        float(
            np.std(
                bandwidth
            )
        ),

        # 46
        float(
            np.mean(
                rolloff
            )
        ),

        # 47
        float(
            np.std(
                rolloff
            )
        ),

        # 48
        float(
            np.mean(
                zcr
            )
        ),

        # 49
        float(
            np.std(
                zcr
            )
        ),

        # 50
        float(
            np.min(
                zcr
            )
        ),

        # 51
        float(
            np.max(
                zcr
            )
        ),

        # 52
        float(
            np.mean(
                rms_frames ** 2
            )
        ),

        # 53
        peak_amplitude,

        # 54
        mean_amplitude,

        # 55
        signal_std,

        # 56
        stft_mean,

        # 57
        stft_std,

        # 58
        float(
            np.mean(
                chroma
            )
        ),

        # 59
        float(
            np.std(
                chroma
            )
        ),

        # 60
        pitch_mean,

        # 61
        pitch_std,

        # 62
        pitch_min,

        # 63
        pitch_max,

        # 64
        duration_seconds,

        # 65
        silence_ratio,

        # 66
        active_ratio,
    ]

    if len(additional_features) != 27:

        raise RuntimeError(
            "Additional acoustic feature block "
            f"produced {len(additional_features)} "
            "features. Expected 27."
        )

    feature_vector.extend(
        additional_features
    )

    # ========================================================
    # FEATURES 67-71
    # 5 SPECTRAL STATISTICS
    # ========================================================

    spectral_profile = np.mean(
        magnitude,
        axis=1
    )

    spectral_statistics = [

        # 67
        float(
            np.mean(
                spectral_profile
            )
        ),

        # 68
        float(
            np.std(
                spectral_profile
            )
        ),

        # 69
        float(
            np.max(
                spectral_profile
            )
        ),

        # 70
        float(
            np.median(
                spectral_profile
            )
        ),

        # 71
        float(
            np.mean(
                spectral_profile ** 2
            )
        ),
    ]

    feature_vector.extend(
        spectral_statistics
    )

    # ========================================================
    # FEATURES 72-75
    # 4 TEMPORAL STATISTICS
    # ========================================================

    temporal_envelope = (
        rms_frames
    )

    temporal_statistics = [

        # 72
        float(
            np.mean(
                temporal_envelope
            )
        ),

        # 73
        float(
            np.std(
                temporal_envelope
            )
        ),

        # 74
        float(
            np.max(
                temporal_envelope
            )
        ),

        # 75
        float(
            np.median(
                temporal_envelope
            )
        ),
    ]

    feature_vector.extend(
        temporal_statistics
    )

    # ========================================================
    # FEATURES 76-79
    # 4 RMS DYNAMICS
    # ========================================================

    rms_dynamics = [

        # 76
        rms_variation,

        # 77
        rms_mean,

        # 78
        rms_min,

        # 79
        rms_max,
    ]

    feature_vector.extend(
        rms_dynamics
    )

    # ========================================================
    # FINAL FEATURE VECTOR
    # ========================================================

    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float64
    )

    feature_vector = np.nan_to_num(
        feature_vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # ========================================================
    # HARD 79-FEATURE CHECK
    # ========================================================

    if len(feature_vector) != 79:

        raise RuntimeError(
            f"Feature extraction produced "
            f"{len(feature_vector)} features. "
            "Expected exactly 79."
        )

    # ========================================================
    # FEATURE SUMMARY
    # ========================================================

    mfcc_variance = float(
        np.mean(
            np.var(
                mfcc,
                axis=1
            )
        )
    )

    delta_mfcc_std = float(
        np.std(
            delta_mfcc
        )
    )

    feature_summary = {

        "mfcc_mean": float(
            np.mean(
                mfcc
            )
        ),

        "mfcc_std": float(
            np.std(
                mfcc
            )
        ),

        "mfcc_variance": mfcc_variance,

        "delta_mfcc_mean": float(
            np.mean(
                delta_mfcc
            )
        ),

        "delta_mfcc_std": delta_mfcc_std,

        "pitch_mean_hz": pitch_mean,

        "pitch_std_hz": pitch_std,

        "pitch_min_hz": pitch_min,

        "pitch_max_hz": pitch_max,

        "pitch_variation_index":
            pitch_variation_index,

        "spectral_centroid_hz":
            float(
                np.mean(
                    centroid
                )
            ),

        "spectral_centroid_std_hz":
            float(
                np.std(
                    centroid
                )
            ),

        "spectral_bandwidth_hz":
            float(
                np.mean(
                    bandwidth
                )
            ),

        "spectral_bandwidth_std_hz":
            float(
                np.std(
                    bandwidth
                )
            ),

        "spectral_rolloff_hz":
            float(
                np.mean(
                    rolloff
                )
            ),

        "spectral_rolloff_std_hz":
            float(
                np.std(
                    rolloff
                )
            ),

        "spectral_flux":
            spectral_flux,

        "zero_crossing_rate":
            float(
                np.mean(
                    zcr
                )
            ),

        "zcr_std":
            float(
                np.std(
                    zcr
                )
            ),

        "rms_mean":
            rms_mean,

        "rms_std":
            rms_variation,

        "rms_min":
            rms_min,

        "rms_max":
            rms_max,

        "silence_ratio":
            silence_ratio,

        "active_ratio":
            active_ratio,

        "duration_seconds":
            duration_seconds,

        "sampling_rate":
            int(sr),

        "peak_amplitude":
            peak_amplitude,

        "mean_amplitude":
            mean_amplitude,

        "signal_std":
            signal_std,

        "chroma_mean":
            float(
                np.mean(
                    chroma
                )
            ),

        "chroma_std":
            float(
                np.std(
                    chroma
                )
            ),

        "stft_mean":
            stft_mean,

        "stft_std":
            stft_std,
    }

    # ========================================================
    # SIGNAL METRICS
    # ========================================================

    signal_metrics = {

        "duration_seconds":
            duration_seconds,

        "sampling_rate":
            int(sr),

        "peak_amplitude":
            peak_amplitude,

        "mean_amplitude":
            mean_amplitude,

        "signal_std":
            signal_std,

        "rms_mean":
            rms_mean,

        "rms_min":
            rms_min,

        "rms_max":
            rms_max,

        "rms_variation":
            rms_variation,

        "zero_crossing_rate":
            float(
                np.mean(
                    zcr
                )
            ),

        "silence_ratio":
            silence_ratio,

        "active_ratio":
            active_ratio,

        "pitch_mean_hz":
            pitch_mean,

        "pitch_std_hz":
            pitch_std,

        "pitch_min_hz":
            pitch_min,

        "pitch_max_hz":
            pitch_max,

        "pitch_variation_index":
            pitch_variation_index,

        "spectral_centroid_hz":
            float(
                np.mean(
                    centroid
                )
            ),

        "spectral_bandwidth_hz":
            float(
                np.mean(
                    bandwidth
                )
            ),

        "spectral_rolloff_hz":
            float(
                np.mean(
                    rolloff
                )
            ),

        "spectral_flux":
            spectral_flux,
    }

    # ========================================================
    # RETURN STRUCTURE
    # ========================================================

    return {

        "feature_vector":
            feature_vector,

        "feature_summary":
            feature_summary,

        "signal_metrics":
            signal_metrics,
    }


# ============================================================
# FILE EXTRACTION
# ============================================================

def extract_from_file(
    file_path
):
    """
    Load audio using SoundFile and extract 79 features.

    No librosa.load().
    """

    file_path = Path(
        file_path
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{file_path}"
        )

    try:

        y, sr = sf.read(
            str(file_path),
            dtype="float32",
            always_2d=False
        )

    except Exception as exc:

        logger.exception(
            "Failed to read audio file."
        )

        raise ValueError(
            "Failed to decode audio file: "
            f"{exc}"
        ) from exc

    return extract_features_from_audio(
        y,
        int(sr)
    )


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_feature_vector(
    feature_vector
):
    """
    Validate the feature vector before
    sending it to the trained model.
    """

    vector = np.asarray(
        feature_vector,
        dtype=np.float64
    ).reshape(-1)

    vector = np.nan_to_num(
        vector,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    if len(vector) != EXPECTED_FEATURE_COUNT:

        raise RuntimeError(
            f"Feature vector has "
            f"{len(vector)} values. "
            f"Expected "
            f"{EXPECTED_FEATURE_COUNT}."
        )

    return vector


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python ml/feature_extraction_test.py "
            "<audio_file>"
        )

        raise SystemExit(1)

    result = extract_from_file(
        sys.argv[1]
    )

    feature_vector = (
        result[
            "feature_vector"
        ]
    )

    print(
        "FEATURE VECTOR SHAPE:",
        feature_vector.shape
    )

    print(
        "FEATURE COUNT:",
        len(feature_vector)
    )

    print(
        "FEATURE EXTRACTION: OK"
    )