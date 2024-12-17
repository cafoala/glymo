import pandas as pd

# Load the cleaned CGM data
df = pd.read_csv("data/processed/3_interpolated_cgm.csv", parse_dates=["time"])

# Ensure `time` is in datetime format
df["time"] = pd.to_datetime(df["time"])

# Create a `date` column for daily grouping
df["date"] = df["time"].dt.date

# Define the minimum threshold for valid readings (90% of 288 = 259)
min_readings = int(288 * 0.9)

# Function to resample per day
def resample_per_day(group):
    # Count only non-NaN glucose values
    non_nan_count = group["glc"].notna().sum()
    
    # Check if the group has at least the minimum number of valid readings
    if non_nan_count < min_readings:
        print(f"Skipping day {group['date'].iloc[0]} for participant {group['ID'].iloc[0]}: {non_nan_count} readings.")
        return pd.DataFrame()  # Skip this day entirely
    
    # Create a full day of 5-minute intervals
    full_day_index = pd.date_range(
        start=f"{group['time'].iloc[0].date()} 00:00:00",
        end=f"{group['time'].iloc[0].date()} 23:55:00",
        freq="5min"
    )
    
    # Resample to ensure exactly 288 readings for the day
    resampled = group.set_index("time").reindex(full_day_index)
    resampled.index.name = "time"
    
    
    # Add back metadata
    resampled["ID"] = group["ID"].iloc[0]
    resampled["date"] = group["date"].iloc[0]
    return resampled.reset_index()

# Apply the function per participant and date
df_resampled = df.groupby(["ID", "date"]).apply(resample_per_day).reset_index(drop=True)

# Save the resampled data
df_resampled.to_csv("data/processed/4_resampled_cgm.csv", index=False)
print("Resampling complete and saved!")
