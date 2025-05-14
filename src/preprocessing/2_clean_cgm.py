import pandas as pd
import numpy as np
import os

# Paths
combined_file = "data/processed/1_combined_cgm.csv"  # Path to the combined file
cleaned_file = "data/processed/2_cleaned_cgm.csv"  # Path to save the cleaned file

# Ensure the output folder exists
os.makedirs(os.path.dirname(cleaned_file), exist_ok=True)

# Load the combined CGM data
print(f"Loading data from: {combined_file}")
df = pd.read_csv(combined_file)

# Ensure `time` is in datetime format
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Round `time` to the nearest 5 minutes
df["time"] = df["time"].dt.round("1min")

# Drop duplicate rows after rounding
df = df.drop_duplicates(subset=["ID", "time"])

# Sort values by ID and time
df = df.sort_values(by=["ID", "time"])

# Ensure glucose values are numeric
df["glc"] = pd.to_numeric(df["glc"], errors="coerce")
#df = df.dropna(subset=["glc"])

# Save the cleaned dataset
df.to_csv(cleaned_file, index=False)

print(f"Cleaning complete! Cleaned data saved to: {cleaned_file}")
