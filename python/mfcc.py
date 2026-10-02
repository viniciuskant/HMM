import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt
import wave
import os
import struct
from config import (SAMPLE_RATE, FRAME_LEN, FRAME_HOP, N_MFCC, N_FFT, N_MELS, USE_DELTAS, PRE_EMPHASIS)

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

def load_audio(audio_path):
    """Step 1: Load the audio file and visualize the waveform"""
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        y = np.frombuffer(wav_file.readframes(n_frames), dtype=np.int16)
    
    frame_size = int(sample_rate * FRAME_LEN)
    frame_step = int(sample_rate * FRAME_HOP)

    return y, sample_rate, frame_size, frame_step

def pre_emphasis(y, pre_emphasis=PRE_EMPHASIS):
    """Step 2: Apply pre-emphasis filter to boost high frequencies"""
    y_preemphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    return y_preemphasized

def frame_signal(y_preemphasized, sample_rate, frame_size, frame_step):
    """Step 3: Frame the signal into overlapping segments"""
    num_samples = len(y_preemphasized)

    num_frames = int(np.ceil(float((num_samples - frame_size -1) / frame_step))) + 1
    
    # Pad signal to ensure all frames have equal number of samples
    pad_signal_length = num_frames * frame_step + frame_size
    z = np.zeros((pad_signal_length - num_samples))
    pad_signal = np.append(y_preemphasized, z)
    
    # Slice the signal into frames
    indices = (np.tile(np.arange(0, frame_size), (num_frames, 1)) + 
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_size, 1)).T)
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames, frame_size

def apply_window(frames, frame_length):
    """Step 4: Apply a window function to each frame"""
    frames *= np.hamming(frame_length)
    return frames

def compute_spectrum(frames, N_FFT=N_FFT):
    """Step 5: Compute the magnitude spectrum of each frame"""
    fft_frames = np.fft.rfft(frames, N_FFT)
    mag_frames = np.absolute(fft_frames)
    pow_frames = np.zeros_like(mag_frames)
    for i in range(len(mag_frames)):
        pow_frames[i] = (mag_frames[i] ** 2) / N_FFT
    return mag_frames, pow_frames

def apply_mel_filterbank(pow_frames, sample_rate, N_FFT=N_FFT, N_MELS=N_MELS):
    """Step 6: Apply the Mel filter bank to the power spectrum"""
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, N_MELS + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((N_FFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((N_MELS, int(np.floor(N_FFT / 2 + 1))))
    for m in range(1, N_MELS + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks <= 0, np.finfo(float).eps, filter_banks)

    filter_banks = 20 * np.log10(filter_banks)
    return filter_banks

def compute_mfcc(filter_banks, N_MFCC=N_MFCC):
    n_frames, n_filters = filter_banks.shape
    mfcc = np.zeros((n_frames, N_MFCC))
    
    for i in range(n_frames):
        x = filter_banks[i]
        
        for k in range(N_MFCC):
            soma = 0.0
            
            for n in range(n_filters):
                cos_val = np.cos(np.pi * (n + 0.5) * k / n_filters)
                mult = x[n] * cos_val
                
                soma += mult
                
            if k == 0:
                escala = np.sqrt(1.0 / n_filters)
            else:
                escala = np.sqrt(2.0 / n_filters)
            
            mfcc[i, k] = soma * escala
            
    
    return mfcc

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

def extract_mfcc(audio_path):
    y, sample_rate, frame_size, frame_step = load_audio(audio_path)
    # print(f"raw: {y[:10]}")
    if len(y) < 400:
        n_feat = N_MFCC * (3 if USE_DELTAS else 1)
        return np.zeros((0, n_feat), dtype=np.float32)

    # y = bandpass_filter(y, sample_rate)
    # print(f"bp: {y[:10]}")
    y_preemphasized = pre_emphasis(y)
    frames, frame_length = frame_signal(y_preemphasized, sample_rate, frame_size, frame_step)
    frames = apply_window(frames, frame_length)
    _, pow_frames = compute_spectrum(frames)
    filter_banks = apply_mel_filterbank(pow_frames, sample_rate)
    mfcc = compute_mfcc(filter_banks)

    # print(f"ANTES do CMVN, frame 0: {mfcc[0]}")

    if USE_DELTAS:
        mfcc_t = mfcc.T
        d1 = librosa.feature.delta(mfcc_t)
        d2 = librosa.feature.delta(mfcc_t, order=2)
        X = np.vstack([mfcc_t, d1, d2]).T
    else:
        X = mfcc

    X = X.astype(np.float32)

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std