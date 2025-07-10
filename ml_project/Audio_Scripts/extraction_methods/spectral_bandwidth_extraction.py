import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.abspath('..'))
from Audio_Scripts import audio_utils as au

def extract_spectral_bandwidth(file_path, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        return np.mean(spectral_bandwidth, axis=1)
    except Exception as e:
        print(f"Error at processing: ❌  {file_path}: {e}")
        return None


def extract_spectral_bandwidth_features(audio_folder, output_csv):
    data = []
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

    for file in tqdm(audio_files):
        file_path = os.path.join(audio_folder, file)
        spectral_bandwidth_features = extract_spectral_bandwidth(file_path)
        if spectral_bandwidth_features is not None:
            gender = au.extract_gender(file)
            student_id = au.extract_student_id(file)
            feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
            for i in range(len(spectral_bandwidth_features)):
                feature_dict[f'bandwidth_{i+1}'] = spectral_bandwidth_features[i]
            data.append(feature_dict)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Spectral bandwidth features saved at {output_csv}")
