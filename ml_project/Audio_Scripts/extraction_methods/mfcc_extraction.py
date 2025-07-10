import librosa
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.insert(1, os.path.abspath('..'))
from Audio_Scripts import audio_utils as au


def extract_mfcc(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y = librosa.util.normalize(y)
        stft = librosa.stft(y, n_fft=512, hop_length=256, win_length=400, window="hann")
        power_spectrogram = np.abs(stft) ** 2
        mel_filterbank = librosa.filters.mel(sr=sr, n_fft=512, n_mels=40)
        mel_spectrogram = np.dot(mel_filterbank, power_spectrogram)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        mfcc = librosa.feature.mfcc(S=log_mel_spectrogram, sr=sr, n_mfcc=n_mfcc)
        mfcc_features = np.mean(mfcc, axis=1)
        return mfcc_features
    except Exception as e:
        print(f"Error at processing: ❌  {file_path}: {e}")
        return None


def extract_mfcc_features(audio_folder, output_csv):
    data = []
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
    for file in tqdm(audio_files):
        try:
            file_path = os.path.join(audio_folder, file)
            mfcc_features = extract_mfcc(file_path)
            if mfcc_features is not None:
                gender = au.extract_gender(file)
                student_id = au.extract_student_id(file)
                feature_dict = {"filename": file, "student_id": student_id, "gender": gender}
                for i in range(len(mfcc_features)):
                    feature_dict[f'mfcc_{i + 1}'] = mfcc_features[i]
                data.append(feature_dict)
        except Exception as e:
            pass
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"MFCC features saved at {output_csv}")
