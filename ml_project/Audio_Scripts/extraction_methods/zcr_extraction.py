import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.abspath('..'))
from Audio_Scripts import audio_utils as au


def extract_zero_crossing_rate(file_path, sr=22050):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        return np.mean(zcr)
    except Exception as e:
        print(f"Error at processing: ❌ {file_path}: {e}")
        return None


def extract_zero_crossing_rate_features(audio_folder, output_csv):
    data = []
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    for file in tqdm(audio_files):
        file_path = os.path.join(audio_folder, file)
        zcr_feature = extract_zero_crossing_rate(file_path)
        if zcr_feature is not None:
            gender = au.extract_gender(file)
            student_id = au.extract_student_id(file)
            data.append({"filename": file, "student_id": student_id, "gender": gender, "zcr": zcr_feature})
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Zero crossing rate features saved at {output_csv}")
