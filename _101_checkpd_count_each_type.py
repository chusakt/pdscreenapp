import os

# Define base path and source folders
from config import base_path

preselect_pd_folder = os.path.join(base_path, "preselectPD")
preselect_control_folder = os.path.join(base_path, "preselectControl")

# List of keywords to search in filenames
keywords = [
    'balance', 'dualTap', 'gaitWalk', 'pinchToSize',
    'questionnaire', 'tremorPostural', 'tremorResting', 'voiceAhh', 'voiceYPL'
]

# Helper function to count files containing specific keywords
def count_files_with_keywords(folder):
    keyword_counts = {keyword: 0 for keyword in keywords}

    for root, _, files in os.walk(folder):
        for file in files:
            for keyword in keywords:
                if keyword in file:
                    keyword_counts[keyword] += 1

    return keyword_counts

# Get counts for 'preselectPD' folder
pd_counts = count_files_with_keywords(preselect_pd_folder)

# Get counts for 'preselectControl' folder
control_counts = count_files_with_keywords(preselect_control_folder)

# Display results neatly
print("File Counts in 'preselectPD' Folder:")
for keyword, count in pd_counts.items():
    print(f"  {keyword}: {count}")

print("\nFile Counts in 'preselectControl' Folder:")
for keyword, count in control_counts.items():
    print(f"  {keyword}: {count}")
