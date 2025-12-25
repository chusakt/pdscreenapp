import os
import json
import csv
import numpy as np
import pandas as pd
import math
from datetime import datetime
from glob import glob

# Define base path and source folders
from config import base_path
preselect_pd_folder = os.path.join(base_path, "preselectPD")
preselect_control_folder = os.path.join(base_path, "preselectControl")

# Define the output CSV file path
output_csv = os.path.join(base_path, "pinch_featureAppdata.csv")

# Function to calculate moving average
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Function to process pinch-like JSON files and compute the required metrics
def process_pinch_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract the subject folder name (immediate folder under preselect_pd or preselect_control)
    subject_folder = os.path.basename(os.path.dirname(os.path.dirname(json_file)))

    # Initialize variables and lists for calculations
    extracted_values = []
    CountAllHandOff = 0
    allSignSwitchCount = 0
    x_mts_start_stack = []
    x_mts_end_stack = []
    x_norm_std_stack = []
    x_norm_mean_stack = []
    x_mx1_std_stack = []
    x_mx1_mean_stack = []
    x_mx2_std_stack = []
    x_mx2_mean_stack = []
    x_my1_std_stack = []
    x_my1_mean_stack = []
    x_my2_std_stack = []
    x_my2_mean_stack = []

    # Check if 'data' exists in the JSON structure
    if 'data' in data:
        for indata in data['data']:
            if 'meta' in indata:  # If 'meta' exists, extract "target" and "absoluteTime"
                j_meta = indata['meta']
                target = j_meta.get('target', None)
                absolute_time = j_meta.get('absoluteTime', None)

                # Initialize variables to store calculations
                mts, mx1, mx2, my1, my2, mdia, mcx, mcy = [], [], [], [], [], [], [], []

                # Check if 'data' key exists next to 'meta'
                if 'data' in indata:
                    j_indata = indata['data']

                    # Perform calculations based on the data
                    for k_injdata in j_indata:
                        CountAllHandOff += 1
                        for m_inkdata in k_injdata:
                            mts.append(m_inkdata['timestamp'])
                            mx1.append(m_inkdata['x1'])
                            mx2.append(m_inkdata['x2'])
                            my1.append(m_inkdata['y1'])
                            my2.append(m_inkdata['y2'])
                            mdia.append(m_inkdata['diameter'])
                            mcx.append(m_inkdata['center']['x'])
                            mcy.append(m_inkdata['center']['y'])

                    # Smoothing and sign-switch counting for mdia
                    mvWindow = 25
                    if len(mdia) > mvWindow + 1:
                        mav = moving_average(mdia, mvWindow)
                        signSwitchCount = 0
                        for idx, dat_ in enumerate(mav):
                            if idx == 0:
                                mvToRaw_a = (dat_ > mdia[idx])
                            else:
                                mvToRaw_b = (dat_ > mdia[idx])
                                if mvToRaw_b != mvToRaw_a:
                                    signSwitchCount += 1
                                    mvToRaw_a = mvToRaw_b
                        allSignSwitchCount += signSwitchCount

                    # Perform normalization and additional calculations
                    mts_ar = np.array(mts)
                    mts_start = np.array(mts[0])
                    mts_end = np.array(mts[-1])

                    mdia_ar = np.array(mdia)
                    if np.max(mdia_ar) != np.min(mdia_ar):  # Ensure denominator is not zero
                        x_norm_ar = (mdia_ar - np.min(mdia_ar)) / (np.max(mdia_ar) - np.min(mdia_ar))
                    else:
                        x_norm_ar = np.zeros_like(mdia_ar)  # If no range, set normalized array to zeros
                    x_norm_std = np.std(x_norm_ar)
                    x_norm_mean = np.mean(x_norm_ar)

                    mdia_ar = np.array(mx1)
                    if np.max(mdia_ar) != np.min(mdia_ar):
                        x_norm_ar = (mdia_ar - np.min(mdia_ar)) / (np.max(mdia_ar) - np.min(mdia_ar))
                    else:
                        x_norm_ar = np.zeros_like(mdia_ar)
                    x_mx1_std = np.std(x_norm_ar)
                    x_mx1_mean = np.mean(x_norm_ar)

                    mdia_ar = np.array(mx2)
                    if np.max(mdia_ar) != np.min(mdia_ar):
                        x_norm_ar = (mdia_ar - np.min(mdia_ar)) / (np.max(mdia_ar) - np.min(mdia_ar))
                    else:
                        x_norm_ar = np.zeros_like(mdia_ar)
                    x_mx2_std = np.std(x_norm_ar)
                    x_mx2_mean = np.mean(x_norm_ar)

                    mdia_ar = np.array(my1)
                    if np.max(mdia_ar) != np.min(mdia_ar):
                        x_norm_ar = (mdia_ar - np.min(mdia_ar)) / (np.max(mdia_ar) - np.min(mdia_ar))
                    else:
                        x_norm_ar = np.zeros_like(mdia_ar)
                    x_my1_std = np.std(x_norm_ar)
                    x_my1_mean = np.mean(x_norm_ar)

                    mdia_ar = np.array(my2)
                    if np.max(mdia_ar) != np.min(mdia_ar):
                        x_norm_ar = (mdia_ar - np.min(mdia_ar)) / (np.max(mdia_ar) - np.min(mdia_ar))
                    else:
                        x_norm_ar = np.zeros_like(mdia_ar)
                    x_my2_std = np.std(x_norm_ar)
                    x_my2_mean = np.mean(x_norm_ar)

                    # Append values to overall stacks
                    x_mts_start_stack.append(mts_start)
                    x_mts_end_stack.append(mts_end)

                    x_norm_std_stack.append(x_norm_std)
                    x_norm_mean_stack.append(x_norm_mean)

                    x_mx1_std_stack.append(x_mx1_std)
                    x_mx1_mean_stack.append(x_mx1_mean)

                    x_mx2_std_stack.append(x_mx2_std)
                    x_mx2_mean_stack.append(x_mx2_mean)

                    x_my1_std_stack.append(x_my1_std)
                    x_my1_mean_stack.append(x_my1_mean)

                    x_my2_std_stack.append(x_my2_std)
                    x_my2_mean_stack.append(x_my2_mean)

    # Final calculations
    mts_start_mean = np.mean(x_mts_start_stack)
    mts_start_max = np.max(x_mts_start_stack)
    mts_range_max = np.max(x_mts_end_stack)

    mdia_std = np.mean(x_norm_std_stack)
    mdia_mean = np.mean(x_norm_mean_stack)

    mx1_std = np.mean(x_mx1_std_stack)
    mx1_mean = np.mean(x_mx1_mean_stack)

    mx2_std = np.mean(x_mx2_std_stack)
    mx2_mean = np.mean(x_mx2_mean_stack)

    my1_std = np.mean(x_my1_std_stack)
    my1_mean = np.mean(x_my1_mean_stack)

    my2_std = np.mean(x_my2_std_stack)
    my2_mean = np.mean(x_my2_mean_stack)

    # Collect the final metrics and return them
    extracted_values.append((
        mdia_std, mdia_mean, mx1_std, mx1_mean,
        mx2_std, mx2_mean, my1_std, my1_mean, my2_std, my2_mean,
        CountAllHandOff, allSignSwitchCount, mts_start_mean, mts_range_max, mts_start_max
    ))
                
    return subject_folder, extracted_values

# Function to calculate age and gender from the Excel file in T1 folder
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

# Function to search and process pinch files in a given folder
def process_folder(base_folder, diagnosis_label, global_index):
    result_list = []
    
    # Search for all JSON files that include 'pinch' in the
    # name
    for json_file in glob(base_folder + '/**/*pinch*.json', recursive=True):
        print(f"Processing file: {json_file}")
        # Extract the subject folder name and computed values from the JSON file
        subject_folder, extracted_values = process_pinch_file(json_file)
        
        # Check for the T1 file to extract age and gender
        t1_files = glob(os.path.join(base_folder, subject_folder, 'T1-*/**/*.xlsx'), recursive=True)
        if t1_files:
            year1, year2, age, gender = extract_age_gender(t1_files[0])
        else:
            year1, year2, age, gender = None, None, None, None

        # Create a row for the current JSON file
        row = [global_index, subject_folder, diagnosis_label, year1, year2, age, gender]

        # Add the computed values to the row
        for values in extracted_values:
            row.extend(values)

        # Append the row to the result list
        result_list.append(row)

        # Increment the global index for the next file
        global_index += 1
    
    return result_list, global_index

# Collect all results from both PD and Control folders
all_results = []
global_index = 1  # Start global index from 1

# Process PD folder and assign diagnosis as 'pd'
pd_results, global_index = process_folder(preselect_pd_folder, 'pd', global_index)
all_results.extend(pd_results)

# Process Control folder and assign diagnosis as 'ct'
ct_results, global_index = process_folder(preselect_control_folder, 'ct', global_index)
all_results.extend(ct_results)

# Write the results to a CSV file with UTF-8 encoding and BOM
with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as csv_file:
    writer = csv.writer(csv_file)
    
    # Write the CSV headers
    writer.writerow([
        'Index', 'Folder Name', 'Diagnosis', 'Timestamp Year', 'Birthdate Year', 'Age', 'Gender', 
        'mdia_std', 'mdia_mean', 'mx1_std', 'mx1_mean', 'mx2_std', 'mx2_mean', 
        'my1_std', 'my1_mean', 'my2_std', 'my2_mean', 'CountAllHandOff', 'allSignSwitchCount', 
        'mts_start_mean', 'mts_range_max', 'mts_start_max'
    ])
    
    # Write all rows
    writer.writerows(all_results)

print(f"Results saved to {output_csv}")

