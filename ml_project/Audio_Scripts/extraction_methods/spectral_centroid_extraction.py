import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.abspath('..'))
from Audio_Scripts import audio_utils as au


def extract_spectral_centroid(file_path, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        return np.mean(spectral_centroid, axis=1)
    except Exception as e:
        print(f"Error at processing: ❌ {file_path}: {e}")
        return None


def extract_spectral_centroids_features(audio_folder, output_csv):
    data = []
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

    for file in tqdm(audio_files):
        file_path = os.path.join(audio_folder, file)
        spectral_centroid_features = extract_spectral_centroid(file_path)
        if spectral_centroid_features is not None:
            gender = au.extract_gender(file)
            student_id = au.extract_student_id(file)
            feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
            for i in range(len(spectral_centroid_features)):
                feature_dict[f'spectral_centroid_{i+1}'] = spectral_centroid_features[i]
            data.append(feature_dict)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Spectral Centroids features saved at {output_csv}")

