import os
import json
import csv
import numpy as np
from glob import glob
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
import pandas as pd
from datetime import datetime
import EntropyHub as EH

# Define base path and source folders
from config import base_path
preselect_pd_folder = os.path.join(base_path, "preselectPD")
preselect_control_folder = os.path.join(base_path, "preselectControl")

# List of interesting files to process
interesting_files_list = ['tremorResting', 'tremorPostural', 'balance','gaitWalk']

# Sampling rate (10 Hz)
sampling_rate = 10.0
nyquist_freq = sampling_rate / 2.0  # Nyquist frequency is half of the sampling rate
cutoff_freq = 0.12  # Desired cutoff frequency (0.12 Hz)
normalized_cutoff = cutoff_freq / nyquist_freq  # Normalized cutoff frequency for the high-pass filter
sos = signal.butter(10, normalized_cutoff, btype='highpass', output='sos')  # Design the high-pass Butterworth filter

# Function to calculate age and gender from the Excel file in T1 folder
def extract_age_gender(t1_file):
    try:
        df = pd.read_excel(t1_file)
        timestamp_col = next((col for col in df.columns if 'วันที่คัดกรอง' in col or 'ประทับเวลา' in col), None)
        timestamp_value = df[timestamp_col].iloc[0] if timestamp_col else None
        birthdate_col = next((col for col in df.columns if 'เกิด(พ.ศ)' in col), None)
        birthdate_value = df[birthdate_col].iloc[0] if birthdate_col else None
        gender_value = df['5. เพศ'].iloc[0] if '5. เพศ' in df.columns else None

        # Extract year values and calculate age
        year1 = pd.to_datetime(timestamp_value, errors='coerce').year if timestamp_value else None
        year2 = pd.to_datetime(birthdate_value, errors='coerce').year if birthdate_value else None
        age = (year1 - year2) if (year1 and year2) else None

        return year1, year2, age, gender_value
    except Exception as e:
        print(f"Error processing file {t1_file}: {e}")
        return None, None, None, None

# Simplified function to process RestHand/PosturalHand/PosturalStability-like JSON files
def process_motion_file(json_file, interesting_files):
    print(f"Processing file: {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
        print(f"Skipping file due to error: {json_file}, Error: {e}")
        return []

    # Extract motion data from 'recording' -> 'recordedData'
    tStamp, acX, acY, acZ, agX, agY, agZ = [], [], [], [], [], [], []
    for i in data['recording']['recordedData']:
        tStamp.append(i['ts'])
        acX.append(i['data'][0])
        acY.append(i['data'][1])
        acZ.append(i['data'][2])
        agX.append(i['data'][3])
        agY.append(i['data'][4])
        agZ.append(i['data'][5])

    # Downsample by selecting actual data points (using decimation)
    downsample_factor = 4
    if len(acX) >= downsample_factor:
        tStamp = signal.decimate(tStamp, downsample_factor, zero_phase=True)
        acX = signal.decimate(acX, downsample_factor, zero_phase=True)
        acY = signal.decimate(acY, downsample_factor, zero_phase=True)
        acZ = signal.decimate(acZ, downsample_factor, zero_phase=True)
        agX = signal.decimate(agX, downsample_factor, zero_phase=True)
        agY = signal.decimate(agY, downsample_factor, zero_phase=True)
        agZ = signal.decimate(agZ, downsample_factor, zero_phase=True)

    print(tStamp[0])
    print(tStamp[-1])
    print(tStamp[-1] - tStamp[0])

    # Handle data selection and resampling based on 'interesting_files'
    if interesting_files in ['RestHand', 'PosturalHand'] and len(acX) > 200:
        acX = signal.resample(acX, 200)
        acY = signal.resample(acY, 200)
        acZ = signal.resample(acZ, 200)
        agX = signal.resample(agX, 200)
        agY = signal.resample(agY, 200)
        agZ = signal.resample(agZ, 200)
    elif interesting_files == 'PosturalStability' and len(acX) > 200:
        mid_index = len(acX) // 2
        start, end = max(0, mid_index - 100), min(len(acX), mid_index + 100)
        acX, acY, acZ = acX[start:end], acY[start:end], acZ[start:end]
        agX, agY, agZ = agX[start:end], agY[start:end], agZ[start:end]
        tStamp = tStamp[start:end]

    # Calculate total time and number of points
    if len(tStamp) > 0:
        total_time = tStamp[-1] - tStamp[0]
        print(f"Total time of data series: {total_time} seconds, Number of points: {len(acX)}")
    else:
        print("No timestamps available.")

    # Feature extraction
    def extract_features(signal_data):
        filtered = signal.sosfilt(sos, signal_data)
        N = len(filtered)
        fft_values = fft(filtered)
        fft_freq = fftfreq(N, 1/sampling_rate)
        magnitude = np.abs(fft_values)
        peak_frequency = fft_freq[np.argmax(magnitude[:N//2])]

        fab = np.abs(fft_values[:N//2])
        F1, F2, F3, F4 = sum(np.square(fab[:25])), sum(np.square(fab[25:50])), sum(np.square(fab[50:75])), sum(np.square(fab[75:]))
        Esum = sum(np.square(fab))
        kur, ske = kurtosis(filtered, fisher=True), skew(filtered, bias=False)
        resdif, resdif2 = np.diff(filtered), np.diff(np.diff(filtered))
        Mobi = np.sqrt(np.var(resdif) / np.var(filtered))
        compx = np.sqrt(np.var(resdif2) * np.var(filtered) / (np.var(resdif) * np.var(resdif)))
        Samp, _, _ = EH.SampEn(filtered, m=2, tau=2)
        E13, E14, E15 = np.percentile(filtered, [25, 50, 75])

        return [np.std(filtered), np.mean(filtered), kur, ske, Mobi, compx,
                F1, F2, F3, F4, F2/Esum, F3/Esum, np.var(resdif2),
                Samp[0], Samp[1], Samp[2], E13, E14, E15, peak_frequency]

    # Extract features for each signal
    features = []
    for signal_data in [acX, acY, acZ, agX, agY, agZ]:
        features.extend(extract_features(signal_data))
    
    return features 

# Function to process files in a given folder for a specific diagnosis
def process_folder(base_folder, diagnosis_label, interesting_files, global_index):
    result_list = []
    for subject_dir in glob(base_folder + '/*'):
        files = glob(subject_dir + f'/**/*{interesting_files}*.json', recursive=True)
        t1_file = glob(subject_dir + '/T1-*/**/*.xlsx', recursive=True)
        year1, year2, age, gender = extract_age_gender(t1_file[0]) if t1_file else (None, None, None, None)

        for idx, file in enumerate(files):
            stats = process_motion_file(file, interesting_files)
            if not stats:
                continue
            
            folder_name = os.path.basename(subject_dir)
            if len(files) > 1:
                folder_name = f"{folder_name}_{idx + 1}"

            result_list.append([global_index, folder_name, year1, year2, age, gender, diagnosis_label] + stats)
            global_index += 1

    return result_list, global_index

# Iterate over each interesting file type
for interesting_files in interesting_files_list:
    print(f"\nProcessing files for: {interesting_files}")

    # Define the output CSV file path
    output_csv = os.path.join(base_path, f"{interesting_files}_featureAppdata.csv")

    # Collect all results from both PD and Control folders
    all_results = []
    global_index = 1

    # Process PD folder and assign diagnosis as 'pd'
    pd_results, global_index = process_folder(preselect_pd_folder, 'pd', interesting_files, global_index)
    all_results.extend(pd_results)

    # Process Control folder and assign diagnosis as 'ct'
    ct_results, global_index = process_folder(preselect_control_folder, 'ct', interesting_files, global_index)
    all_results.extend(ct_results)

    # Define the feature names
    def get_feature_names():
        base_features = ['std', 'mean', 'kurtosis', 'skewness', 'mobility', 'complexity',
                         'F1', 'F2', 'F3', 'F4', 'F2/Esum', 'F3/Esum', 'var_resdif2',
                         'SampEn0', 'SampEn1', 'SampEn2', 'Percentile_25', 'Percentile_50', 'Percentile_75', 'peak_frequency']
        signals = ['acX', 'acY', 'acZ', 'agX', 'agY', 'agZ']
        headers = [f'{signal}_{feature}' for signal in signals for feature in base_features]
        return headers

    # Write the results to a CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        headers = ['Index', 'Folder Name', 'Timestamp Year', 'Birthdate Year', 'Age', 'Gender', 'Diagnosis'] + get_feature_names()
        writer.writerow(headers)
        writer.writerows(all_results)

    print(f"Results saved to {output_csv}")
