import parselmouth
from parselmouth.praat import call
import statistics
import numpy as np
from scipy.stats.mstats import zscore
import os
import pickle

from config import base_path
modeltouse = "trained_model_voice_Ahh_SMOTE.pkl"
new_file_path_modeltouse = os.path.join(base_path, modeltouse)

# Load the trained model from the pickle file
with open(new_file_path_modeltouse, 'rb') as f:
    model = pickle.load(f)

# Function to measure pitch using the given method
def measurePitch(wav_file, f0min, f0max, unit):
    sound = parselmouth.Sound(wav_file)  # Read the sound
    duration = call(sound, "Get total duration")  # Duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)  # Create a Praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit)  # Get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit)  # Get standard deviation
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

    return [
        duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, 
        rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, 
        apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer
    ]

# Function to measure formants
def measureFormants(wav_file, f0min, f0max):
    sound = parselmouth.Sound(wav_file)  # Read the sound
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list, f2_list, f3_list, f4_list = [], [], [], []

    for point in range(numPoints):
        t = call(pointProcess, "Get time from index", point + 1)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

    f1_list = [f for f in f1_list if not np.isnan(f)]
    f2_list = [f for f in f2_list if not np.isnan(f)]
    f3_list = [f for f in f3_list if not np.isnan(f)]
    f4_list = [f for f in f4_list if not np.isnan(f)]

    f1_median = statistics.median(f1_list) if f1_list else None
    f2_median = statistics.median(f2_list) if f2_list else None
    f3_median = statistics.median(f3_list) if f3_list else None
    f4_median = statistics.median(f4_list) if f4_list else None

    f1_mean = statistics.mean(f1_list) if f1_list else None
    f2_mean = statistics.mean(f2_list) if f2_list else None
    f3_mean = statistics.mean(f3_list) if f3_list else None
    f4_mean = statistics.mean(f4_list) if f4_list else None

    return [f1_median, f2_median, f3_median, f4_median, f1_mean, f2_mean, f3_mean, f4_mean]

# Function to calculate additional features from formant medians
def calculate_additional_features(f1_median, f2_median, f3_median, f4_median):
    formants = [f1_median, f2_median, f3_median, f4_median]
    formants_zscore = zscore(formants) if None not in formants else [None] * 4

    # pF: z-scored formants
    pF = sum(formants_zscore) / 4 if None not in formants_zscore else None

    # fdisp: Formant dispersion
    fdisp = (f4_median - f1_median) / 3 if f1_median and f4_median else None

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

# Example usage with a single WAV file
wav_file_path = "./datajson/control/ขจร เขียวหวาน_1721110619936_voiceAhh.wav"  # Replace with the path to your WAV file

# Measure pitch and formants
# try:
pitch_features = measurePitch(wav_file_path, f0min=75, f0max=300, unit="Hertz")
formant_medians_means = measureFormants(wav_file_path, f0min=75, f0max=300)

# Split formant medians and means
formant_medians = formant_medians_means[:4]
formant_means = formant_medians_means[4:]

# Calculate additional features
additional_features = calculate_additional_features(*formant_medians)

# Prepare to print all features
pitch_feature_names = [
    "Duration", "Mean F0", "Stdev F0", "HNR", "Local Jitter", "Local Absolute Jitter", 
    "RAP Jitter", "PPQ5 Jitter", "DDP Jitter", "Local Shimmer", "Local dB Shimmer", 
    "APQ3 Shimmer", "APQ5 Shimmer", "APQ11 Shimmer", "DDA Shimmer"
]
formant_feature_names = [
    "F1 Median", "F2 Median", "F3 Median", "F4 Median", 
    "F1 Mean", "F2 Mean", "F3 Mean", "F4 Mean"
]
additional_feature_names = [
    "pF", "fdisp", "avgFormant", "mff", "fitch_vtl", "delta_f", "vtl_delta_f", "xysum"
]

# Merge all extracted features into a single list
all_features = pitch_features + formant_medians + formant_means + additional_features

# Combine all feature names into a single list
all_feature_names = (
    pitch_feature_names + 
    formant_feature_names + 
    additional_feature_names
)

# Print all features together
print("All Extracted Features:")
for name, value in zip(all_feature_names, all_features):
    print(f"  {name}: {value}")

# --------------------------------------------
# Convert dictionary values to a flat list
all_features = all_features[1:]

# Convert to a 2D NumPy array
X_test = np.array(all_features).reshape(1, -1)

# Predict on the test set
y_pred = model.predict(X_test)

# print output compare to target

print('predicted:',y_pred)

# except Exception as e:
#     print(f"Error processing file {wav_file_path}: {e}")



