# Script for processing Weinstock (2016)
# Author: Sangaman Senthil
# Date: February 5th, 2020, edited June 14th, by Elizabeth Chun
# Adjusted for directory updates and compatibility

import os
import pandas as pd

# Paths
raw_folder = "data/raw/Weinstock2016"  # Path to raw dataset
processed_folder = "data/processed/Weinstock2016"  # Path to save processed data
os.makedirs(processed_folder, exist_ok=True)  # Ensure processed folder exists

# File paths
input_file = os.path.join(raw_folder, "Data Tables", "BDataCGM.txt")
output_file = os.path.join(processed_folder, "Weinstock2016_processed.csv")

# Read raw data
print(f"Reading file: {input_file}")
df = pd.read_csv(input_file, sep="|", low_memory=False)

# Drop unwanted columns
columns_to_drop = ['RecID', 'BCGMDeviceType', 'BFileType', 'CalBG']
df = df.drop(columns=columns_to_drop)

# Rename columns to standard names
df = df.rename(columns={
    'PtID': 'ID',
    'DeviceDaysFromEnroll': 'time',
    'DeviceTm': 'tm',
    'Glucose': 'glc'
})

# Meeting format standards
# Add a fixed base date to the 'time' column (assumes 'time' is days since a start date)
base_date = "1990-01-01"
df['time'] = pd.to_datetime(base_date) + pd.to_timedelta(df['time'], unit='D')

# Combine date and time columns into one datetime column
df['time'] = df['time'].dt.strftime('%Y-%m-%d') + " " + df['tm']

# Drop the 'tm' column since it's now combined into 'time'
df = df.drop(columns=['tm'])

# Export the final processed dataset
print(f"Saving processed data to: {output_file}")
df.to_csv(output_file, index=False)

print("Processing complete!")
