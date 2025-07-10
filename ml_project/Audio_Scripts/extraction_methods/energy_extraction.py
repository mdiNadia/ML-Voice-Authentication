import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.abspath('..'))
from Audio_Scripts import audio_utils as au


def extract_energy(file_path, sr=22050, hop_length=512):
    try:
        y, sr = librosa.load(file_path, sr=sr)   # Read Audio File
        y = librosa.util.normalize(y)            # Normalization
        rms_energy = librosa.feature.rms(y=y, hop_length=hop_length)  # Root Mean Square (RMS) Energy
        # Generate time vector based on number of frames
        times = librosa.frames_to_time(np.arange(rms_energy.shape[1]), sr=sr, hop_length=hop_length)
        return rms_energy, times
    except Exception as e:
        print(f"Error While processing ❌  {file_path}: {e}")
        return None, None


def extract_energy_features(audio_folder, output_csv):
    data = []
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    for file in tqdm(audio_files):
        file_path = os.path.join(audio_folder, file)
        rms_energy, _ = extract_energy(file_path)
        if rms_energy is not None:
            gender = au.extract_gender(file)
            student_id = au.extract_student_id(file)
            energy_features = np.mean(rms_energy, axis=1)
            feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
            for i in range(len(energy_features)):
                feature_dict[f'energy_{i + 1}'] = energy_features[i]
            data.append(feature_dict)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Energy features saved at {output_csv}")
