import os
import csv
import statistics
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment
import numpy as np
from glob import glob
# from scipy.stats import zscore
from scipy.stats.mstats import zscore
import pandas as pd
from datetime import datetime

# Define base path and source folders
from config import base_path
# preselect_pd_folder = os.path.join(base_path, "preselectPD")
# preselect_control_folder = os.path.join(base_path, "preselectControl")
preselect_pd_folder = os.path.join(base_path, "preselectPD")
preselect_control_folder = os.path.join(base_path, "preselectControl")
# Define the output CSV file path
output_csv = os.path.join(base_path, "voice_featureAppdata.csv")

# Convert mp4 to wav using pydub
def convert_to_wav(file_path):
    if file_path.endswith(".wav"):
        return file_path
    elif file_path.endswith(".mp4") or file_path.endswith(".m4a"):
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.replace(".mp4", ".wav").replace(".m4a", ".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    else:
        print(f"Unsupported file format: {file_path}")
        return None


# Function to measure pitch using the given method
def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) # create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return [duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter,
            localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer]

# Function to measure formants
# Function to measure formants
def measureFormants(voiceID, f0min, f0max):
    sound = parselmouth.Sound(voiceID)  # read the sound
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []

    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

    f1_list = [f1 for f1 in f1_list if not np.isnan(f1)]
    f2_list = [f2 for f2 in f2_list if not np.isnan(f2)]
    f3_list = [f3 for f3 in f3_list if not np.isnan(f3)]
    f4_list = [f4 for f4 in f4_list if not np.isnan(f4)]

    f1_median = statistics.median(f1_list) if f1_list else None
    f2_median = statistics.median(f2_list) if f2_list else None
    f3_median = statistics.median(f3_list) if f3_list else None
    f4_median = statistics.median(f4_list) if f4_list else None

    # Calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list) if f1_list else None
    f2_mean = statistics.mean(f2_list) if f2_list else None
    f3_mean = statistics.mean(f3_list) if f3_list else None
    f4_mean = statistics.mean(f4_list) if f4_list else None

    return [f1_median, f2_median, f3_median, f4_median, f1_mean, f2_mean, f3_mean, f4_mean]


# Function to extract age and gender from the T1 file
def extract_age_gender(t1_file):
    try:
        df = pd.read_excel(t1_file)
        
        # Extract 'ประทับเวลา', 'วัน/เดือน/ปีเกิด', and 'เพศ'
        timestamp_value = df['ประทับเวลา'].iloc[0] if 'ประทับเวลา' in df.columns else None
        birthdate_value = df['วัน/เดือน/ปีเกิด'].iloc[0] if 'วัน/เดือน/ปีเกิด' in df.columns else None
        gender_value = df['5. เพศ'].iloc[0] if '5. เพศ' in df.columns else None
        
        # Extract only the year
        if timestamp_value is not None and isinstance(timestamp_value, datetime):
            year1 = timestamp_value.year
        elif timestamp_value is not None:
            year1 = pd.to_datetime(str(timestamp_value), format='%d/%m/%Y', errors='coerce').year
        else:
            year1 = None
        
        if birthdate_value is not None and isinstance(birthdate_value, datetime):
            year2 = birthdate_value.year
        elif birthdate_value is not None:
            year2 = pd.to_datetime(str(birthdate_value), format='%d/%m/%Y', errors='coerce').year
        else:
            year2 = None
        
        # Calculate age
        if year1 is not None and year2 is not None:
            age = year1 - year2
            if age < 0:
                year2 -= 543  # Convert Buddhist year to Gregorian year
                age = year1 - year2
        else:
            age = None
        
        # Return the values and calculated age along with gender
        return year1, year2, age, gender_value

    except Exception as e:
        print(f"Error processing file {t1_file}: {e}")
        return None, None, None, None


# Function to calculate additional features from formant medians
def calculate_additional_features(f1_median, f2_median, f3_median, f4_median):
    formants = [f1_median, f2_median, f3_median, f4_median]
    # print (formants )
    formants_zscore = zscore(formants) if None not in formants else [None] * 4

    # pF: z-scored formants
    pF = sum(formants_zscore) / 4 if None not in formants_zscore else None
    # pF = sum(zscore(f1_median),zscore(f2_median),zscore(f3_median),zscore(f4_median) ) / 4
    
    # fdisp: Formant dispersion
    fdisp = (f4_median - f1_median) / 3 if f1_median and f4_median else None
    # fdisp = (f4_median - f1_median) / 3 

    # avgFormant: Average formants
    avgFormant = sum(formants) / 4 if None not in formants else None

    # mff: Multiplicative formant frequency
    mff = (f1_median * f2_median * f3_median * f4_median) ** 0.25 if None not in formants else None

    # fitch_vtl: Fitch's vocal tract length estimation
    fitch_vtl = ((1 * (35000 / (4 * f1_median))) +
                 (3 * (35000 / (4 * f2_median))) +
                 (5 * (35000 / (4 * f3_median))) +
                 (7 * (35000 / (4 * f4_median)))) / 4 if None not in formants else None

    # delta_f: Delta F calculation
    xysum = (0.5 * f1_median) + (1.5 * f2_median) + (2.5 * f3_median) + (3.5 * f4_median) if None not in formants else None
    xsquaredsum = (0.5 ** 2) + (1.5 ** 2) + (2.5 ** 2) + (3.5 ** 2)
    delta_f = xysum / xsquaredsum if xysum else None

    # vtl_delta_f: Vocal tract length from delta_f
    vtl_delta_f = 35000 / (2 * delta_f) if delta_f else None

    return [pF, fdisp, avgFormant, mff, fitch_vtl, delta_f, vtl_delta_f, xysum]







# Modified function to search for filenames containing 'ahh' or 'Yai' within 'T4' subfolders and flatten files
# def search_files_in_T4_and_analyze(base_folder, diagnosis_label, global_index):
import os
from glob import glob

def search_files_in_T4_and_analyze(base_folder, diagnosis_label, global_index):
    result_list = []

    # Search for files in all subdirectories of base_folder recursively
    for subject_folder in glob(os.path.join(base_folder, '**/'), recursive=True):
        print(f"Processing subject folder: {subject_folder}")

        # Look for specific audio files in subfolders (recursive)
        matching_files = glob(os.path.join(subject_folder, '*Ahh*.*'), recursive=True) + \
                         [file for file in glob(os.path.join(subject_folder, '*YPL*.*'), recursive=True) if 'Situa' not in file]

        # Search for 'T1' subfolder to extract age and gender
        t1_folders = glob(os.path.join(subject_folder, '**/T1*/'), recursive=True)
        if t1_folders:
            try:
                t1_file = glob(os.path.join(t1_folders[0], '*.xlsx'))[0]  # Assumes there's an Excel file in the T1 folder
                year1, year2, age, gender = extract_age_gender(t1_file)
            except IndexError:
                print(f"No valid T1 file found in {t1_folders[0]}")
                year1, year2, age, gender = None, None, None, None
        else:
            print(f"No T1 folder found in {subject_folder}")
            year1, year2, age, gender = None, None, None, None

        for file_path in matching_files:
            filename = os.path.basename(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))
            print(f"Found file: {filename}")

            # Convert to WAV and check if conversion is successful
            wav_file = convert_to_wav(file_path)
            if wav_file is None:
                continue

            # Measure pitch and formants
            try:
                pitch_features = measurePitch(wav_file, f0min=75, f0max=300, unit="Hertz")
                formant_medians_means = measureFormants(wav_file, f0min=75, f0max=300)

                formant_medians = formant_medians_means[:4]
                formant_means = formant_medians_means[4:]

                additional_features = calculate_additional_features(*formant_medians)
            except Exception as e:
                print(f"Error processing file {wav_file}: {e}")
                pitch_features = [None] * 15
                formant_medians = [None] * 4
                formant_means = [None] * 4
                additional_features = [None] * 8

            # Append the features to the result list, including age and gender
            result_list.append([global_index, folder_name, filename, diagnosis_label, year1, year2, age, gender] +
                               pitch_features + formant_medians + formant_means + additional_features)
            global_index += 1

    return result_list, global_index




# Collect all results from both PD and Control folders
all_results = []
global_index = 1  # Start global index from 1

# Process PD folder and assign diagnosis as 'pd'
pd_results, global_index = search_files_in_T4_and_analyze(preselect_pd_folder, 'pd', global_index)
all_results.extend(pd_results)

# Process Control folder and assign diagnosis as 'ct'
control_results, global_index = search_files_in_T4_and_analyze(preselect_control_folder, 'ct', global_index)
all_results.extend(control_results)


# # Define the headers for the CSV output
# headers = ['Index', 'Subject Folder Name', 'Filename', 'Diagnosis', 'Timestamp Year', 'Birth Year', 'Age', 'Gender',
#            'Duration', 'Mean F0', 'Stdev F0', 'HNR', 'Local Jitter', 'Local Absolute Jitter', 
#            'RAP Jitter', 'PPQ5 Jitter', 'DDP Jitter', 'Local Shimmer', 'Local dB Shimmer', 
#            'APQ3 Shimmer', 'APQ5 Shimmer', 'APQ11 Shimmer', 'DDA Shimmer', 
#            'F1 Median', 'F2 Median', 'F3 Median', 'F4 Median',
#            'pF', 'fdisp', 'avgFormant', 'mff', 'fitch_vtl', 'delta_f', 'vtl_delta_f']

# Define the headers for the CSV output, adding the mean formant columns
headers = ['Index', 'Folder Name', 'Filename', 'Diagnosis', 'Timestamp Year', 'Birth Year', 'Age', 'Gender',
           'Duration', 'Mean F0', 'Stdev F0', 'HNR', 'Local Jitter', 'Local Absolute Jitter', 
           'RAP Jitter', 'PPQ5 Jitter', 'DDP Jitter', 'Local Shimmer', 'Local dB Shimmer', 
           'APQ3 Shimmer', 'APQ5 Shimmer', 'APQ11 Shimmer', 'DDA Shimmer', 
           'F1 Median', 'F2 Median', 'F3 Median', 'F4 Median',  # Median Formants
           'F1 Mean', 'F2 Mean', 'F3 Mean', 'F4 Mean',           # Mean Formants
           'pF', 'fdisp', 'avgFormant', 'mff', 'fitch_vtl', 'delta_f', 'vtl_delta_f', 'xysum']  # Added xysum



# Write the results to a CSV file with UTF-8 encoding and BOM
with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as csv_file:
    writer = csv.writer(csv_file)

    # Write the CSV headers
    writer.writerow(headers)

    # Write all rows
    writer.writerows(all_results)

print(f"Results saved to {output_csv}")

