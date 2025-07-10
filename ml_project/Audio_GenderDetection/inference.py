import os
import sys
import shutil

import glob
import joblib
import librosa
import pandas as pd
import soundfile as sf
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.insert(1, os.path.abspath(os.path.join(os.path.abspath("."), "..")))

from Audio_Scripts import preprocessing
from Audio_Scripts.extraction_methods import mfcc_extraction
from Audio_Scripts.extraction_methods import energy_extraction
from Audio_Scripts.extraction_methods import log_mel_spectrogram_extraction
from Audio_Scripts.extraction_methods import spectral_bandwidth_extraction
from Audio_Scripts.extraction_methods import spectral_contrast_extraction
from Audio_Scripts.extraction_methods import spectral_centroid_extraction
from Audio_Scripts.extraction_methods import zcr_extraction

samping_rate = 22050
hop_length = 512
lowcut, highcut = 50.0, 5000.0
segment_duration = 3


def detect_gender(audio_path, model):
    # Load Model and scaler
    if isinstance(model, str):
        model = joblib.load(model)
    if not hasattr(model, "predict"):
        raise ValueError(f"Model {model} not supported")

    if not audio_path.endswith(".wav") and not audio_path.endswith(".mp3"):
        raise ValueError(f"Audio path {audio_path} not supported")

    scaler = joblib.load(os.path.join("saved_models", "std_scaler.joblib"))

    temp_dir = "./.temp.inference_cache"
    os.makedirs(temp_dir, exist_ok=True)

    # Split audio into 3 seconds chunks (maximum 1 min)
    audio, sr = librosa.load(audio_path, sr=samping_rate)
    filtered_audio = preprocessing.bandpass_filter(audio, sr, lowcut, highcut)
    normalized_audio = preprocessing.normalize_audio(filtered_audio, sr, target_lufs=-14)
    non_silent_audio = preprocessing.remove_silence(normalized_audio, threshold=0.05)
    if non_silent_audio.size <= 0:
        raise ValueError(f"file `{audio_path}` is a silent file.")

    num_samples_per_segment = sr * segment_duration
    total_segments = min(5, len(normalized_audio) // num_samples_per_segment)  # maximum 15 seconds per person
    for i in range(total_segments):
        start = i * num_samples_per_segment
        end = start + num_samples_per_segment
        segment = normalized_audio[start:end]
        segment_file = os.path.join(temp_dir, f"segment_{i}.wav")
        sf.write(segment_file, segment, sr)

    # Extract features and run model
    chunks = [os.path.join(temp_dir, f"segment_{i}.wav") for i in range(total_segments)]
    results = []
    for chunk_name in tqdm(chunks):
        energy, _ = energy_extraction.extract_energy(chunk_name, sr=samping_rate, hop_length=hop_length)
        energy_features = list(np.mean(energy, axis=1))
        mfcc_features = mfcc_extraction.extract_mfcc(chunk_name, n_mfcc=13)
        mfcc_features = list(mfcc_features)
        spectral_contrast = spectral_contrast_extraction.extract_spectral_contrast(audio_path, sr=samping_rate)
        spectral_contrast_features = list(spectral_contrast)
        spectral_bandwidth = spectral_bandwidth_extraction.extract_spectral_bandwidth(audio_path, sr=samping_rate)
        spectral_bandwidth_features = list(spectral_bandwidth)
        mel_spectrogram = log_mel_spectrogram_extraction.extract_log_mel_spectrogram(audio_path, n_mels=40,
                                                                                     n_fft=1024, hop_length=hop_length)
        mel_spectrogram_features = list(mel_spectrogram)
        spectral_centroid = spectral_centroid_extraction.extract_spectral_centroid(audio_path, sr=samping_rate)
        spectral_centroid_features = list(spectral_centroid)
        zero = zcr_extraction.extract_zero_crossing_rate(audio_path, sr=samping_rate)
        zero_features = [zero]

        all_features = energy_features + mfcc_features + spectral_contrast_features + spectral_bandwidth_features + mel_spectrogram_features + spectral_centroid_features + zero_features
        all_features = np.array(all_features)

        features_ready = scaler.transform([all_features])
        prediction = model.predict(features_ready)
        results.append(list(prediction)[0])
    shutil.rmtree(temp_dir)
    return 1 if sum(results) > len(results) // 2 else 0


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input audio file')
    parser.add_argument(
        '-m', "--model", type=str,
        default=os.path.join("saved_models", "support_vector_machine_model.joblib"),
        help='Model to used for inference.')
    args = parser.parse_args()

    print(detect_gender(args.input, args.model))