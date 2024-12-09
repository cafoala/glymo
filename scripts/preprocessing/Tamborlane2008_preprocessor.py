import os
import pandas as pd

# Paths
dataset = "data/raw/Tamborlane2008"
file_path = os.path.join(dataset, "DataTables")
output_file = "data/processed/Tamborlane2008/processed_cgm.csv"

# Ensure the processed directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# List only the files matching the pattern
files = [f for f in os.listdir(file_path) if "RTCGM" in f]

# Loop through the files and process
for i, filename in enumerate(files):
    full_path = os.path.join(file_path, filename)
    print(f"Processing file {i + 1}/{len(files)}: {full_path}")

    # Read the CSV file
    curr = pd.read_csv(full_path)

    # Remove unnecessary columns
    if "RecID" in curr.columns:
        curr.drop(columns=["RecID"], inplace=True)

    # Rename columns to standard names
    curr.columns = ["ID", "time", "glc"]

    # Convert datetime to standardized format
    try:
        #curr["time"] = pd.to_datetime(curr["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        curr["time"] = pd.to_datetime(curr["time"], errors="coerce")
    except Exception as e:
        print(f"Datetime parsing failed for {filename}: {e}")
        continue

    # Debug: Check rows with missing datetimes
    missing_times = curr[curr["time"].isna()]
    print(f"Rows with missing time: {missing_times.shape[0]}")

    # Drop rows where datetime conversion failed
    curr = curr.dropna(subset=["time"])

    # Ensure glucose values are numeric
    curr["glc"] = pd.to_numeric(curr["glc"], errors="coerce")

    # Append to the output file
    curr.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)

print(f"Processing complete! Processed data saved to: {output_file}")
