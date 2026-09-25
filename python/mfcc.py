import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
from config import (SAMPLE_RATE, FRAME_LEN, FRAME_HOP, N_MFCC, N_FFT, N_MELS, USE_DELTAS)

# Limites do filtro (Hz)
HIGHPASS_HZ = 80.0
LOWPASS_HZ = 7000.0
FILTER_ORDER = 4


def bandpass_filter(y: np.ndarray, sr: int) -> np.ndarray:
    nyq = sr / 2.0
    low_n = max(HIGHPASS_HZ / nyq, 1e-4)
    high_n = min(LOWPASS_HZ / nyq, 0.999)
    sos = butter(FILTER_ORDER, [low_n, high_n], btype="band", output="sos")
    return sosfiltfilt(sos, y).astype(np.float32)


def extract_mfcc_from_array(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if len(y) < 400:
        return np.zeros((0, N_MFCC * (3 if USE_DELTAS else 1)), dtype=np.float32)

    y = bandpass_filter(y, sr)
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    #MFCC
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=int(FRAME_HOP * sr),
        win_length=int(FRAME_LEN * sr),
        n_mels=N_MELS, window="hamming",
    )

    # deltas
    feats = [mfcc]
    if USE_DELTAS:
        feats = [mfcc, librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)]

    X = np.vstack(feats).T.astype(np.float32)

    #CMVN
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std


def extract_mfcc(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return extract_mfcc_from_array(y, SAMPLE_RATE)