import os
import json
import csv
import numpy as np
import math
import pandas as pd
from datetime import datetime
from glob import glob
from statistics import mean

# Define base path and source folders
from config import base_path

preselect_pd_folder = os.path.join(base_path, "preselectPD")
preselect_control_folder = os.path.join(base_path, "preselectControl")

# Define the output CSV file path
output_csv = os.path.join(base_path, "dualtap_featureAppdata.csv")

# Function to extract TapCount and additional statistics from the JSON file
def extract_tapcount_statistics(json_file):
    print(f"Processing file: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    score = data['score']

    # --------------  about circle --------------- 
    jL = data['recording']['circleL']
    jR = data['recording']['circleR']
    cLx, cLy = jL['x'], jL['y']
    cRx, cRy = jR['x'], jR['y']
    cRadius = jL['r']

    distancEachPoint = []
    inLeft = 0
    inRight = 0
    jf = data['recording']['taps']
    ts_instroke_mean = []

    for jstroke in jf:
        jjj = jstroke['data']
        ts_instroke = []
        for jjjj in jjj:
            xpo, ypo, ts_ = jjjj['x'], jjjj['y'], jjjj['ts']
            ts_instroke.append(ts_)

            # Compare to L and R, then decide which side
            if (xpo - cLx) >= (cRx - xpo):
                inRight += 1
                dispo = math.sqrt((xpo - cRx) ** 2 + (ypo - cRy) ** 2)
                distancEachPoint.append(dispo)
            else:
                inLeft += 1
                dispo = math.sqrt((xpo - cLx) ** 2 + (ypo - cLy) ** 2)
                distancEachPoint.append(dispo)

        ts_instroke_mean.append(mean(ts_instroke))

    countinside = sum(map(lambda x: x < cRadius, distancEachPoint))
    countall = len(distancEachPoint)
    ppInsideToAll = countinside / countall if countall > 0 else 0

    try:
        if inLeft / inRight > 1:
            ppLeftToRight = inRight / inLeft
        else:
            ppLeftToRight = inLeft / inRight
    except ZeroDivisionError:
        ppLeftToRight = 0

    tDiff = [item1 - item2 for item1, item2 in zip(ts_instroke_mean[1:], ts_instroke_mean[:-1])]
    tDiff_mean = mean(tDiff) if tDiff else 0
    tDiff_max = max(tDiff) if tDiff else 0
    tDiff_min = min(tDiff) if tDiff else 0

    tap_points_list = [len(tap['data']) for tap in jf]
    maxTapPoints = np.max(tap_points_list) if tap_points_list else 0
    stdTapPoints = np.std(tap_points_list) if tap_points_list else 0
    meanTapPoints = np.mean(tap_points_list) if tap_points_list else 0
    tapCount = len(tap_points_list)

    return (score, tapCount, maxTapPoints, stdTapPoints, meanTapPoints, countall, 
            ppInsideToAll, ppLeftToRight, tDiff_mean, tDiff_max, tDiff_min)

# Function to extract 'ประทับเวลา', 'วัน/เดือน/ปีเกิด', 'เพศ' and calculate 'Age'
def extract_timestamp_age_and_gender(excel_file):
    try:
        df = pd.read_excel(excel_file)
        
        # Extract 'ประทับเวลา', 'วัน/เดือน/ปีเกิด', and 'เพศ'
        timestamp_value = df['ประทับเวลา'].iloc[0] if 'ประทับเวลา' in df.columns else None
        birthdate_value = df['วัน/เดือน/ปีเกิด'].iloc[0] if 'วัน/เดือน/ปีเกิด' in df.columns else None
        gender_value = df['5. เพศ'].iloc[0] if '5. เพศ' in df.columns else None
        
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
        
        if year1 is not None and year2 is not None:
            age = year1 - year2
            if age < 0:
                year2 -= 543
                age = year1 - year2
        else:
            age = None

        return year1, year2, age, gender_value

    except Exception as e:
        print(f"Error processing file {excel_file}: {e}")
        return None, None, None, None

# Function to search and process dualtap files in a given folder
def process_folder(base_folder, diagnosis_label, global_index):
    result_list = []
    
    # Search for all JSON files that include 'dualtap' in the name
    for subject_dir in glob(base_folder + '/*'):
        dualtap_files = glob(subject_dir + '/**/*dualtap*.json', recursive=True)
        t1_file = glob(subject_dir + '/T1-*/**/*.xlsx', recursive=True)
        year1, year2, age, gender = extract_timestamp_age_and_gender(t1_file[0]) if t1_file else (None, None, None, None)
        
        for idx, dualtap_file in enumerate(dualtap_files):
            print(f"Processing dualtap file: {dualtap_file}")
            stats = extract_tapcount_statistics(dualtap_file)

            folder_name = os.path.basename(subject_dir)
            if len(dualtap_files) > 1:
                folder_name = f"{folder_name}_{idx + 1}"
            
            result_list.append([global_index, folder_name, year1, year2, age, gender, diagnosis_label] + list(stats))
            global_index += 1

    return result_list, global_index

# Collect all results from both PD and Control folders
all_results = []
global_index = 1

# Process PD folder and assign diagnosis as 'pd'
pd_results, global_index = process_folder(preselect_pd_folder, 'pd', global_index)
all_results.extend(pd_results)

# Process Control folder and assign diagnosis as 'ct'
ct_results, global_index = process_folder(preselect_control_folder, 'ct', global_index)
all_results.extend(ct_results)

# Write the results to a CSV file
with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Index', 'Folder Name', 'Timestamp Year', 'Birthdate Year', 'Age', 'Gender', 'Diagnosis', 'Score', 
                     'Tap Count', 'Max Tap Points', 'Std Tap Points', 'Mean Tap Points', 'Total Points', 
                     'ppInsideToAll', 'ppLeftToRight', 'tDiff_mean', 'tDiff_max', 'tDiff_min'])
    writer.writerows(all_results)

print(f"Results saved to {output_csv}")
