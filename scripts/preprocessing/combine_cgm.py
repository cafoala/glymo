import os
import pandas as pd

# Define directories
standardized_folder = "data/processed/cgm/"  # Folder containing standardized files
output_file = "data/processed/combined_cgm.csv"  # Output combined file

# Ensure output folder exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# List all CSV files in the standardized folder
files = [f for f in os.listdir(standardized_folder) if f.endswith(".csv")]

# Initialize an empty list to store DataFrames
dataframes = []

# Process each file
for filename in files:
    # Full path to the current file
    file_path = os.path.join(standardized_folder, filename)

    # Read the file
    df = pd.read_csv(file_path)

    # Rename columns to standard names
    df.rename(columns={"id": "ID", "gl": "glc"}, inplace=True)

    # Extract dataset name (before `_`) from the filename
    dataset_name = filename.split("_")[0]

    # Update the ID column to include dataset name
    if "ID" in df.columns:
        df["ID"] = dataset_name + "_" + df["ID"].astype(str)

    # Select only the relevant columns
    df = df[["ID", "time", "glc"]]

    # Append the DataFrame to the list
    dataframes.append(df)

    print(f"Processed file: {filename}")

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a CSV file
combined_df.to_csv(output_file, index=False)

print(f"Combined CGM data saved to: {output_file}")
