import os
import librosa
from tqdm import tqdm
import re

def extract_gender(filename: str):
    """Find the speaker gender of an audio file by looking at its filename."""
    if "female" in filename.lower():
        return "female"
    elif "male" in filename:
        return "male"
    else:
        return "unknown"

def extract_student_id(filename: str):
    """Find the student ID of an audio file by looking at its filename. Student id is a 9-digit number inside filename."""
    pattern = r'\d{9}'
    match = re.search(pattern, filename)
    if match:
        return match.group(0)
    return -1

def get_audio_info(file_path: str):
    """This function is used to extract the information from the audio file. This information includes:
    1.Gender , 2.NumberOfChannels, 3.Sampling rate, 4.Duration."""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=False)
        if y.ndim == 1:
            num_channels = 1
        else:
            num_channels = y.shape[0]
        duration = librosa.get_duration(y=y, sr=sr)
        gender = extract_gender(file_path)
        student_id = extract_student_id(file_path)
        return {
            "num_channels": num_channels,
            "duration": duration,
            "sampling_rate": sr,
            "student_id": student_id,
            "gender": gender
        }
    except Exception as e:
        print(f"Error reading the audio file: {e}")
        return None


def get_audio_info_from_files(files: list):
    """Extract the general information of files inside a folder."""
    total_duration, male_duration, female_duration, unknown_duration = 0, 0, 0, 0
    sampling_rates, num_channels_list = [], []
    n_males, n_females, n_unknown = 0, 0, 0
    n_distinct_speakers = {"males": {}, "females": {}, "unknown": {}}

    for path in tqdm(files):
        audio_info = get_audio_info(path)
        if audio_info:
            total_duration += audio_info['duration']
            if audio_info['gender'] == 'male':
                male_duration += audio_info['duration']
                n_males += 1
                if audio_info["student_id"] in n_distinct_speakers["males"]:
                    n_distinct_speakers["males"][audio_info["student_id"]] += 1
                else:
                    n_distinct_speakers["males"][audio_info["student_id"]] = 1
            elif audio_info['gender'] == 'female':
                female_duration += audio_info['duration']
                n_females += 1
                if audio_info["student_id"] in n_distinct_speakers["females"]:
                    n_distinct_speakers["females"][audio_info["student_id"]] += 1
                else:
                    n_distinct_speakers["females"][audio_info["student_id"]] = 1
            else:
                unknown_duration += audio_info['duration']
                n_unknown += 1
                if audio_info["student_id"] in n_distinct_speakers["unknown"]:
                    n_distinct_speakers["unknown"][audio_info["student_id"]] += 1
                else:
                    n_distinct_speakers["unknown"][audio_info["student_id"]] = 1
            sampling_rates.append(audio_info['sampling_rate'])
            num_channels_list.append(audio_info['num_channels'])

    return {
        "total_duration": total_duration,
        "male_duration": male_duration,
        "female_duration": female_duration,
        "unknown_duration": unknown_duration,
        "sampling_rates": sampling_rates,
        "num_channels": num_channels_list,
        "n_males": n_males,
        "n_females": n_females,
        "n_unknowns": n_unknown,
        "n_speakers": n_distinct_speakers,
    }


def extract_speakers_id(filenames: list):
    """This function finds all the unique speakers inside raw data and return their audio filenames."""
    males, females, unknowns, bad = dict(), dict(), dict(), list()
    for audio in tqdm(filenames):
        try:
            student_id = extract_student_id(audio)
            gender = extract_gender(audio)
            if gender == "male":
                if student_id not in males:
                    males[student_id] = [audio]
                else:
                    males[student_id].append(audio)
            elif gender == "female":
                if student_id not in females:
                    females[student_id] = [audio]
                else:
                    females[student_id].append(audio)
            else:
                if student_id not in unknowns:
                    females[student_id] = [audio]
                else:
                    females[student_id].append(audio)
        except Exception as e:
            bad.append(audio)
    return males, females, unknowns, bad


def raw_audio_files(directory: str):
    """This function only return the file names of the audio files. Files are divided into two set:
    1. Bad files are those which the gender or student is not present in filename.
    2. Good files are files that includes student id and gender within filename."""
    bad_files, good_files = [], []
    for audio in tqdm(os.listdir(directory)):
        student_id = extract_student_id(audio)
        if student_id == -1:
            bad_files.append(os.path.join(directory, audio))
        elif "male" not in audio.lower():
            bad_files.append(os.path.join(directory, audio))
        else:
            good_files.append(os.path.join(directory, audio))
    return bad_files, good_files


def convert_filenames_to_lowercase(directory):
    """After looking at filenames, we found out that some of the audio files includes capital letters. So it is better
     to convert all the filenames into lowercase to use our regular functions on filenames."""
    try:
        for filename in os.listdir(directory):
            old_file_path = os.path.join(directory, filename)
            if os.path.isfile(old_file_path):  # Check if it's a file
                new_filename = filename.lower()
                new_file_path = os.path.join(directory, new_filename)
                new_file_path = new_file_path.replace("-", "_")
                new_file_path = new_file_path.replace("_student", "")
                os.rename(old_file_path, new_file_path)
        print("All filenames have been converted to lowercase.")
    except Exception as e:
        print(f"An error occurred: {e}")