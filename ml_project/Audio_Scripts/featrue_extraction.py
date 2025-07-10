"""This module is used to extract features from audio files"""
import glob
import pandas as pd
import sys

from extraction_methods import mfcc_extraction
from extraction_methods import energy_extraction
from extraction_methods import log_mel_spectrogram_extraction
from extraction_methods import spectral_bandwidth_extraction
from extraction_methods import spectral_contrast_extraction
from extraction_methods import spectral_centroid_extraction
from extraction_methods import zcr_extraction

import os


def extract_features_from_audios(directory: str):
    print("1/7: Extracting energy features... ")
    csv_file = os.path.join(directory, 'energy_features.csv')
    energy_extraction.extract_energy_features(directory, csv_file)
    print("-" * 20)

    print("2/7: Extracting mfcc features... ")
    csv_file = os.path.join(directory, 'mfcc_features.csv')
    mfcc_extraction.extract_mfcc_features(directory, csv_file)
    print("-" * 20)

    print("3/7: Extracting spectral contrast features... ")
    csv_file = os.path.join(directory,'spectral_contrast_features.csv')
    spectral_contrast_extraction.extract_spectral_contrast_features(directory, csv_file)
    print("-" * 20)

    print("4/7: Extracting spectral bandwidth features... ")
    csv_file = os.path.join(directory, 'spectral_bandwidth_features.csv')
    spectral_bandwidth_extraction.extract_spectral_bandwidth_features(directory, csv_file)
    print("-" * 20)

    print("5/7: Extracting log mel spectrogram features... ")
    csv_file = os.path.join(directory, 'log_mel_features.csv')
    log_mel_spectrogram_extraction.extract_log_mel_spectrogram_features(directory, csv_file)
    print("-" * 20)

    print("6/7: Extracting spectral centroid features... ")
    csv_file = os.path.join(directory, 'spectral_centroid_features.csv')
    spectral_centroid_extraction.extract_spectral_centroids_features(directory, csv_file)
    print("-" * 20)

    print("7/7: Extracting zcr features... ")
    csv_file = os.path.join(directory,  'zcr_features.csv')
    zcr_extraction.extract_zero_crossing_rate_features(directory, csv_file)
    print("-" * 20)

    # ---------- Merge all feature ----------- #
    feature_files = glob.glob(os.path.join(directory, "*.csv"))
    feature_files = [f for f in feature_files if 'train.csv' not in f]
    if not feature_files:
        raise FileNotFoundError("No Feature files found.")
    df_list = [pd.read_csv(file) for file in feature_files if 'features.csv' in file]
    df_final = df_list[0]
    for df in df_list[1:]:
        df_final = pd.merge(df_final, df, on=['filename', 'student_id', 'gender'], how='left', suffixes=('', '_dup'))
    df_final = df_final.loc[:, ~df_final.columns.str.endswith('_dup')]
    df_final.to_csv(os.path.join(directory, "all_features.csv"), index=False)

    return os.path.join(directory, "all_features.csv")
