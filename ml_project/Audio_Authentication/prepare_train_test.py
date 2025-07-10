"""Prepare train and test data for voice authentication."""


import numpy as np
import os
import sys
import random
import shutil
import librosa
sys.path.insert(1, "..")
sys.path.insert(1, os.path.join("..", 'Audio_Scripts'))
import soundfile as sf
from Audio_Scripts import audio_utils as au
from tqdm import tqdm


n_replica = 3
raw_data = os.path.join("..", "Data", "raw")

bad_files, good_files = au.raw_audio_files(raw_data)
males, females, unknowns, bad = au.extract_speakers_id(good_files)

# Remove those speakers who have less than 3 audio files
males_speakers_reduced, females_speakers_reduces = {}, {}
for speaker, files in males.items():
    if len(files) >= 3:
        males_speakers_reduced[speaker] = files
for speaker, files in females.items():
    if len(files) >= 3:
        females_speakers_reduces[speaker] = files


print("Total number of males speaker:", len(males))
print("Total number of females speaker:", len(females))

males_speaker, females_speakers = set(males_speakers_reduced.keys()), set(females_speakers_reduces.keys())
all_speakers = {}
all_speakers.update(males_speakers_reduced)
all_speakers.update(females_speakers_reduces)

# Randomly select 6 students and copy their voice in Authentication directory.
for i in range(n_replica):
    selected_males_speaker = set(random.choices(list(males_speaker), k=3))
    selected_females_speaker = set(random.choices(list(females_speakers), k=3))
    selected_speakers = selected_males_speaker | selected_females_speaker
    while len(selected_speakers) < 6:
        selected_males_speaker = set(random.choices(list(males_speaker), k=3))
        selected_females_speaker = set(random.choices(list(females_speakers), k=3))
        selected_speakers = selected_males_speaker | selected_females_speaker

    if os.path.exists(os.path.join(f"data_{i}", 'train')):
        shutil.rmtree(os.path.join(f"data_{i}", 'train'))

    for j, speaker in enumerate(selected_speakers):
        os.makedirs(os.path.join(f"data_{i}", 'train', str(j)))
        for z, audio in enumerate(all_speakers[speaker]):
            shutil.copyfile(audio, os.path.join(f"data_{i}", 'train', str(j), f'{z}.mp3'))


def split_audio(audio_file, duration, max_chunk=10_000, sr=16000):
    audio_data, sr = librosa.load(audio_file, sr=sr)
    num_samples_per_segment = sr * duration
    total_segments = min(max_chunk, len(audio_data) // num_samples_per_segment)
    for seg in range(total_segments):
        start = seg * num_samples_per_segment
        end = start + num_samples_per_segment
        segment = audio_data[start:end]
        segment_file = os.path.join(os.path.dirname(audio_file),
                                    f"{os.path.basename(audio_file).split('.')[0]}_{seg}.wav")
        sf.write(segment_file, segment, sr)


# split audios into 3 seconds chunks
print("Splitting audios into 3 seconds chunks...")
for n in range(n_replica):
    train_data_folder = os.path.join(f"data_{n}", 'train')
    for j in tqdm(range(6)):
        data_folder = os.path.join(train_data_folder, str(j))
        files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
        for f in files:
            split_audio(f, 3)
            os.remove(f)

# Separate test data
print("Separating test data from train data")
test_ratio = 0.25
for i in range(n_replica):
    train_data_folder = os.path.join(f"data_{i}", 'train')
    for j in tqdm(range(6)):
        data_folder = os.path.join(train_data_folder, str(j))
        list_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)]
        test_size = int(test_ratio * len(list_files))
        test_files = random.choices(list_files, k=test_size)
        test_folder = os.path.join(f"data_{i}", "test", str(j))
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)
        os.makedirs(test_folder)
        for f in test_files:
            try:
                shutil.copy(f, os.path.join(test_folder, os.path.basename(f)))
                os.remove(f)
            except Exception:
                pass


def augment_audio(audio, sr):
    pitch_shifted = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=np.random.randint(-2, 3))
    time_stretched = librosa.effects.time_stretch(y=audio, rate=np.random.uniform(0.8, 1.2))
    return pitch_shifted, time_stretched


def generate_augmented_audios(train_folder, n):
    for class_folder in range(0, 6):
        print("Augmenting data from class: ", class_folder)
        class_path = os.path.join(train_folder, str(class_folder))
        audio_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        current_count = len(audio_files)
        if current_count >= n:
            print(f"Class {class_folder} already has {current_count} files, skipping augmentation.")
            continue

        while current_count < n:
            for audio_file in audio_files:
                audio_path = os.path.join(class_path, audio_file)
                audio, sr = librosa.load(audio_path, sr=None)
                augmented_audios = augment_audio(audio, sr)
                for idx, aug_audio in enumerate(augmented_audios):
                    new_file_name = f"{os.path.splitext(audio_file)[0]}_aug_{current_count + idx + 1}.wav"
                    new_file_path = os.path.join(class_path, new_file_name)
                    sf.write(new_file_path, aug_audio, sr)
                    #print(f"Saved augmented audio: {new_file_path}")
                current_count += len(augmented_audios)
                if current_count >= n:
                    break
            audio_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]

    print("Augmentation complete.")


for i in range(3):
    train_folder = f'data_{i}/train'
    n = 1000
    generate_augmented_audios(train_folder, n)