import pandas as pd

# Paths
cleaned_file = "data/processed/2_cleaned_cgm.csv"
interpolated_file = "data/processed/3_interpolated_cgm.csv"

# Function to interpolate
def interpolate_per_id(group):
    # Resample only numeric columns (e.g., glucose values)
    numeric_cols = ["glc"]
    group_numeric = group.set_index("time")[numeric_cols].resample("5min").mean()

    # Add back the ID column
    group_numeric["ID"] = group["ID"].iloc[0]

    # Interpolate glucose values
    group_numeric["glc"] = group_numeric["glc"].interpolate(method="pchip", limit=4, limit_direction="forward").round(1)
    
    # Reset index to include 'time' as a column
    return group_numeric.reset_index()

# Chunk processing with groupby
chunk_size = 500000  # Adjust chunk size
header_written = False

for chunk in pd.read_csv(cleaned_file, chunksize=chunk_size, parse_dates=["time"]):
    print(f"Processing chunk with {len(chunk)} rows...")
    chunk = chunk.sort_values(["ID", "time"])  # Ensure data is sorted by ID and time

    # Apply interpolation per ID
    interpolated_chunk = chunk.groupby("ID", group_keys=False).apply(interpolate_per_id)
    
    # Save results
    interpolated_chunk.to_csv(interpolated_file, mode="a", index=False, header=not header_written)
    header_written = True

print(f"Interpolation complete! Results saved to: {interpolated_file}")
