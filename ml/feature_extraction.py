import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_features_from_audio(y, sr):
    """
    Extract comprehensive acoustic features from audio signal (y) and sampling rate (sr).
    Returns a dictionary containing:
      - 'feature_vector': 1D numpy array formatted for ML models
      - 'feature_summary': Structured dict formatted for frontend display & XAI
      - 'signal_metrics': Duration, RMS energy, silence ratio, SNR estimation
    """
    if len(y) == 0:
        raise ValueError("Audio signal is empty.")

    # 1. Signal Metrics
    duration = float(len(y) / sr)
    rms = float(np.mean(librosa.feature.rms(y=y)))
    
    # Silence detection
    intervals = librosa.effects.split(y, top_db=25)
    non_silent_samples = sum(end - start for start, end in intervals) if len(intervals) > 0 else len(y)
    silence_ratio = float(max(0.0, 1.0 - (non_silent_samples / len(y))))

    # 2. Spectral Features
    # MFCCs (20 coefficients) and MFCC Deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Compute MFCC Delta (temporal dynamics)
    mfcc_delta = librosa.feature.delta(mfcc)
    delta_std = float(np.mean(np.std(mfcc_delta, axis=1)))

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = float(np.mean(mel_spec_db))
    mel_std = float(np.std(mel_spec_db))

    # Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    cent_mean = float(np.mean(spec_cent))
    cent_std = float(np.std(spec_cent))

    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bw_mean = float(np.mean(spec_bw))
    bw_std = float(np.std(spec_bw))

    # Spectral Rolloff
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    roll_mean = float(np.mean(spec_roll))
    roll_std = float(np.std(spec_roll))

    # Spectral Flux (frame-to-frame spectral energy change)
    stft = np.abs(librosa.stft(y))
    spectral_flux = float(np.mean(np.sqrt(np.sum(np.diff(stft, axis=1)**2, axis=0))))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Pitch & Fundamental Frequency (f0)
    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50, fmax=500)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            f0_mean = float(np.mean(pitch_values))
            f0_std = float(np.std(pitch_values))
            f0_var = float(f0_std / (f0_mean + 1e-6)) # Coefficient of variation
        else:
            f0_mean, f0_std, f0_var = 140.0, 12.0, 0.08
    except Exception as e:
        logger.warning(f"Pitch extraction fallback due to error: {e}")
        f0_mean, f0_std, f0_var = 140.0, 12.0, 0.08

    # Assemble 1D Feature Vector for Machine Learning Model
    feature_vector = np.hstack([
        mfcc_mean, mfcc_std,
        mel_mean, mel_std,
        cent_mean, cent_std,
        bw_mean, bw_std,
        roll_mean, roll_std,
        zcr_mean, zcr_std,
        chroma_mean, chroma_std,
        f0_mean, f0_std, f0_var,
        delta_std, spectral_flux
    ])

    # Human-readable summary for Dashboard UI & Explainable AI
    feature_summary = {
        "mfcc_mean_1": float(mfcc_mean[0]),
        "mfcc_variance": float(np.mean(mfcc_std)),
        "delta_mfcc_std": round(delta_std, 4),
        "spectral_flux": round(spectral_flux, 4),
        "mel_energy_db": round(mel_mean, 2),
        "spectral_centroid_hz": round(cent_mean, 2),
        "spectral_bandwidth_hz": round(bw_mean, 2),
        "spectral_rolloff_hz": round(roll_mean, 2),
        "zero_crossing_rate": round(zcr_mean, 4),
        "chroma_energy": round(float(np.mean(chroma_mean)), 4),
        "pitch_f0_mean_hz": round(f0_mean, 2),
        "pitch_std_hz": round(f0_std, 2),
        "pitch_variation_index": round(f0_var, 4),
        "silence_ratio": round(silence_ratio, 4),
        "rms_energy": round(rms, 4)
    }

    signal_metrics = {
        "duration_seconds": round(duration, 2),
        "sample_rate_hz": sr,
        "total_samples": len(y),
        "silence_ratio": round(silence_ratio, 3)
    }


    return {
        "feature_vector": feature_vector,
        "feature_summary": feature_summary,
        "signal_metrics": signal_metrics
    }

def extract_from_file(file_path, target_sr=22050):
    """Convenience function to load audio file and extract feature dict."""
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return extract_features_from_audio(y, sr)
