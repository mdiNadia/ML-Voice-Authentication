import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.abspath('..'))
from Audio_Scripts import audio_utils as au


def extract_spectral_contrast(file_path, sr=22050, n_bands=6):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=n_bands)
        return np.mean(spectral_contrast, axis=1)
    except Exception as e:
        print(f"Error at processing: ❌  {file_path}: {e}")
        return None


def extract_spectral_contrast_features(audio_folder, output_csv):
    data = []
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    for file in tqdm(audio_files):
        file_path = os.path.join(audio_folder, file)
        spectral_contrast_features = extract_spectral_contrast(file_path)
        if spectral_contrast_features is not None:
            gender = au.extract_gender(file)
            student_id = au.extract_student_id(file)
            feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
            for i in range(len(spectral_contrast_features)):
                feature_dict[f'spectral_contrast_{i+1}'] = spectral_contrast_features[i]
            data.append(feature_dict)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Spectral contrast features saved at {output_csv}")
