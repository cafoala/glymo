import os
import pandas as pd

# Define directories
standardized_folder = "data/processed/cgm/"  # Folder containing standardized files
output_file = "data/processed/1_combined_cgm.csv"

# Ensure the output folder exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Initialize an empty list to store DataFrames
dataframes = []

# Define standard column names
standard_columns = ["ID", "time", "glc"]

# Process each file
for filename in os.listdir(standardized_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(standardized_folder, filename)
        print(f"Processing file: {filename}")
        
        # Extract the prefix from the filename (everything before the first "_")
        prefix = filename.split("_")[0] + "_"
        
        # Read the file
        df = pd.read_csv(file_path)
        
        # Rename columns to standard names
        df.rename(columns={"id": "ID", "gl": "glc"}, inplace=True)
        
        if prefix in ['dexi_', 'dexip_', 'extodedu_', 'extod101_']:
            print(prefix+ ' is in the list')
            df['glc'] = df['glc'] * 18

        # Ensure the file has the required columns
        missing_columns = [col for col in standard_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"File {filename} is missing columns: {missing_columns}")
        
        # Add the prefix to the ID column
        df["ID"] = prefix + df["ID"].astype(str)
        
        # Reorder columns to standard order
        df = df[standard_columns]
        
        # Add the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame
combined_df.to_csv(output_file, index=False)
print(f"Combined CGM data saved to: {output_file}")
