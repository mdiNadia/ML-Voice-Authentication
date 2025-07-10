import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.abspath('..'))
from Audio_Scripts import audio_utils as au


def extract_log_mel_spectrogram(file_path, sr=22050, n_mels=40, n_fft=1024, hop_length=512):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        y = librosa.util.normalize(y)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return np.mean(log_mel_spectrogram, axis=1)
    except Exception as e:
        print(f"Error at processing: ❌  {file_path}: {e}")
        return None


def extract_log_mel_spectrogram_features(audio_folder, output_csv):
    data = []
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    for file in tqdm(audio_files):
        file_path = os.path.join(audio_folder, file)
        log_mel_features = extract_log_mel_spectrogram(file_path)
        if log_mel_features is not None:
            gender = au.extract_gender(file)
            student_id = au.extract_student_id(file)
            feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
            for i in range(len(log_mel_features)):
                feature_dict[f'log_mel_{i+1}'] = log_mel_features[i]
            data.append(feature_dict)
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Log Mel Spectrogram features saved at {output_csv}")
