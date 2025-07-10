import os
import glob
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pyloudnorm import Meter, normalize
from scipy import signal
from audio_utils import convert_filenames_to_lowercase


def bandpass_filter(audio, sr, lowcut, highcut, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='bandpass')
    return signal.filtfilt(b, a, audio)


def remove_silence(audio_data, threshold=0.01, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    frames = np.where(rms > threshold)[0]
    if len(frames) > 0:
        return np.concatenate([audio_data[i * hop_length:(i + 1) * hop_length] for i in frames])
    else:
        return audio_data


# Normalize audio
def normalize_audio(audio_data, sr, target_lufs=-14):
    meter = Meter(sr)
    loudness = meter.integrated_loudness(audio_data)
    return normalize.loudness(audio_data, loudness, target_lufs)


# Convert label names into numbers
def label_to_number(label):
    if "female.mp3" in label.lower():
        return 0
    elif "male.mp3" in label.lower():
        return 1
    print(f"Warning: Unknown label for file {label}")
    return -1


def process_file(input_file, output_dir, output_data, lowcut=50.0, highcut=5000.0, sr=22050, segment_duration=3):
    try:
        audio, sr = librosa.load(input_file, sr=sr)
        filtered_audio = bandpass_filter(audio, sr, lowcut, highcut)
        normalized_audio = normalize_audio(filtered_audio, sr, target_lufs=-14)
        non_silent_audio = remove_silence(normalized_audio, threshold=0.05)
        if non_silent_audio.size > 0:

            label = label_to_number(os.path.basename(input_file))
            num_samples_per_segment = sr * segment_duration
            total_segments = min(100, len(normalized_audio) // num_samples_per_segment)  # maximum 5 min per person

            for i in range(total_segments):
                start = i * num_samples_per_segment
                end = start + num_samples_per_segment
                segment = normalized_audio[start:end]
                segment_file = os.path.join(output_dir,
                                            f"{os.path.basename(input_file).split('.')[0]}_segment_{i}.wav")
                sf.write(segment_file, segment, sr)
                output_data.append([label, segment_file])
            pass
        else:
            print(f"Skipped: {input_file} (no non-silent audio detected)")
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")


# Process all files inside a folder.
def process_directory(input_directory_, output_directory_, output_csv_, lowcut=50.0, highcut=5000.0, sr=22050,
                      segment_duration=3):
    #convert_filenames_to_lowercase(input_directory_)
    audio_files = (glob.glob(os.path.join(input_directory_, "*.mp3")) +
                   glob.glob(os.path.join(input_directory_, "*.wav")))

    print(f"Found {len(audio_files)} inside {input_directory_}")
    if not os.path.exists(output_directory_):
        os.makedirs(output_directory_)

    if not os.path.exists(os.path.dirname(output_csv_)):
        os.makedirs(os.path.dirname(output_csv_))

    output_data = []
    for i, audio_file in enumerate(audio_files):
        audio_file = audio_file.replace("Male", "male").replace("MALE", "male")
        print(f"{i+1}/{len(audio_files)}-Processing file: {audio_file}")
        if "male" not in audio_file:
            print(f"Skipped: {audio_file} , gender is unknown")
            continue
        process_file(audio_file, output_directory_, output_data, lowcut, highcut, sr, segment_duration)

    if output_data:
        df = pd.DataFrame(output_data, columns=['label', 'segment_file'])
        df.to_csv(output_csv_, index=False)
        print(f"Saved segments metadata to {output_csv_}")
    else:
        print("No data to save. No valid segments found.")
