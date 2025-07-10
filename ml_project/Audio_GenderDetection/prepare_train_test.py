""" This module is used to generate train and test split for Gender detection task. Since most of the audios
are male's audio, we must ba careful about male and female audio ratio.
"""

import os
import sys
import random
import shutil
import pandas as pd
from typing import Union
sys.path.insert(1, '..')
sys.path.insert(1, os.path.join("..", 'Audio_Scripts'))
from Audio_Scripts import audio_utils as au
from Audio_Scripts import preprocessing
from Audio_Scripts import featrue_extraction


# ------------- Setup and calculate required variables -------------- #
test_ratio = 0.25
m2f_ratio_train: Union[None, int] = 1
m2f_ratio_test = 1
# ------------------------------------------------------------------- #
print("Preparing dataset for gender detection ... ")

# --------------- Read Raw Data File Names -------------------------- #
raw_data_path = os.path.join("..", "Data", "raw")
print(" >>> Converting filenames into lower case for better analysis ...")
preprocessing.convert_filenames_to_lowercase(raw_data_path)
print("\n >>> Separate good filenames and bad filenames ... ")
bad_files, filenames = au.raw_audio_files(raw_data_path)
print(f"\nNumber of total audio files: {len(bad_files) + len(filenames)} [Bad Format: {len(bad_files)}] "
      f"[Correct format: {len(filenames)}] [Since Now, We would only work on good files!]")
print("*"*40)

# ----------------- Extract general information from good data ----------- #
print(f" >>> Extracting general information of audio files with correct format...")
raw_data_info = au.get_audio_info_from_files(filenames)
print(" >>> Total duration: ", raw_data_info["total_duration"] / 3600)
print(" >>> Male duration: ", raw_data_info["male_duration"] / 3600)
print(" >>> Female duration: ", raw_data_info["female_duration"] / 3600)
print(" >>> Unknown gender duration: ", raw_data_info["unknown_duration"] / 3600)
print()
print(" >>> Number of men audio files: ", raw_data_info["n_males"])
print(" >>> Number of women audios files: ", raw_data_info["n_females"])
print(" >>> Number of unknown audios files: ", raw_data_info["n_unknowns"])
print()
n_distinct_males = len(raw_data_info["n_speakers"]["males"])
n_distinct_females = len(raw_data_info["n_speakers"]["females"])
n_distinct_unknowns = len(raw_data_info["n_speakers"]["unknown"])
print(" >>> Number of distinct speakers: ", n_distinct_males + n_distinct_females + n_distinct_unknowns)
print(" >>> Number of distinct male speakers: ", n_distinct_males)
print(" >>> Number of distinct female speakers: ", n_distinct_females)
print(" >>> Number of distinct unknown speakers: ", n_distinct_unknowns)
print("*"*40)

# ------------ Calculate required number of files for train and test set ----------- #
print("\n >>> Calculating the required number of train and test speakers ...")
num_test_speakers = int((n_distinct_males + n_distinct_females) * test_ratio)
n_females_speakers_test = int(num_test_speakers / (m2f_ratio_test + 1))
n_males_speaker_test = num_test_speakers - n_females_speakers_test

num_train_speakers = (n_distinct_males + n_distinct_females) - num_test_speakers
n_females_speakers_train = n_distinct_females - n_females_speakers_test
n_males_speaker_train = n_distinct_males - n_males_speaker_test

#if m2f_ratio_train is not None:
#    n_females_speakers_train = int(num_train_speakers / (m2f_ratio_train + 1))
#    n_males_speaker_train = num_train_speakers - n_females_speakers_train

if m2f_ratio_train is not None:
    n_males_speaker_train = n_females_speakers_train

if n_females_speakers_train + n_females_speakers_test > n_distinct_females:
    n_females_speakers_train = n_distinct_females - n_females_speakers_train
    n_males_speaker_train = n_females_speakers_train

if n_males_speaker_train + n_males_speaker_test > n_distinct_males:
    n_males_train = n_distinct_males - n_males_speaker_test
    n_males_speaker_train = n_males_speaker_test

print("\n >>> Male speakers in train data: ", n_males_speaker_train)
print(" >>> Female speakers in train data: ", n_females_speakers_train)
print(" >>> Male speakers in test data: ", n_males_speaker_test)
print(" >>> Female speakers in test data: ", n_females_speakers_test)
print("-"*40)


# ------------------- Split dataset into train and test ---------------- #
print("\n >>> Separating train and test speakers for males and females...")
males, females, unknowns, bad = au.extract_speakers_id(filenames)
train_males_speaker, train_females_speakers = set(males.keys()), set(females.keys())

test_males_speaker = set(random.choices(list(train_males_speaker), k=n_males_speaker_test))
test_females_speaker = set(random.choices(list(train_females_speakers), k=n_females_speakers_test))
train_males_speaker.difference_update(test_males_speaker)
train_females_speakers.difference_update(test_females_speaker)

train_males_speaker = set(random.choices(list(train_males_speaker), k=n_males_speaker_train))
train_females_speakers = set(random.choices(list(train_females_speakers), k=n_females_speakers_train))

train_speakers = train_males_speaker.union(train_females_speakers)
test_speakers = test_males_speaker.union(test_females_speaker)

train_path_raw = os.path.join('.', 'train', 'raw')
test_path_raw = os.path.join('.', 'test', 'raw')

if os.path.exists(train_path_raw):
    shutil.rmtree(train_path_raw)
if os.path.exists(test_path_raw):
    shutil.rmtree(test_path_raw)

os.makedirs(train_path_raw)
os.makedirs(test_path_raw)

print(" >>> Saving raw audios ... ")
for m_speaker in train_males_speaker:
    for file_ in males[m_speaker]:
        shutil.copyfile(file_, os.path.join(train_path_raw, os.path.basename(file_)))
for f_speaker in train_females_speakers:
    for file_ in females[f_speaker]:
        shutil.copyfile(file_, os.path.join(train_path_raw, os.path.basename(file_)))

n_test_per_person = 2
for m_speaker in test_males_speaker:
    for i, file_ in enumerate(males[m_speaker]):
        shutil.copyfile(file_, os.path.join(test_path_raw, os.path.basename(file_)))
        if i >= n_test_per_person -1:
            break
for f_speaker in test_females_speaker:
    for i, file_ in enumerate(females[f_speaker]):
        shutil.copyfile(file_, os.path.join(test_path_raw, os.path.basename(file_)))
        if i >= n_test_per_person -1:
            break
print(" >>> Train and Test data stored in ./train and ./test")
print("-"*40)

# ------------------ Preprocess selected audios ----------------- #
print("\n >>> Processing selected audios for training")
train_path_processed = os.path.join('.', 'train', 'processed')
train_path_raw = os.path.join('.', 'train', 'raw')
output_csv = os.path.join(train_path_processed, 'train.csv')
preprocessing.process_directory(train_path_raw, train_path_processed, output_csv)


# --------------------- Feature extraction --------------- #
print("-"*40)
print("\n >>> Extracting features for train data...")
train_path_processed = os.path.join('.', 'train', 'processed')
features_files = featrue_extraction.extract_features_from_audios(train_path_processed)
print()
print(f" >>> All training features saved at: {features_files}")
print("-"*40)

features = pd.read_csv(features_files)
features['label'] = features['gender'].apply(lambda x: 1 if x == 'male' else 0)
features = features.drop(columns=['gender', 'filename', 'student_id'])
features.to_csv(os.path.join(".", "train", "train_features.csv"), index=False)
print(f' >>> Train features are stored in {os.path.join(".", "train", "train_features.csv")}')
