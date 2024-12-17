import numpy as np
import pandas as pd

# Load the normalized data
df = pd.read_csv("data/processed/5_normalized_cgm.csv")

# Ensure `time` is in datetime format
df["time"] = pd.to_datetime(df["time"])

# Create a `date` column for daily grouping
df["date"] = df["time"].dt.date

# Validate daily counts
daily_counts = df.groupby(["ID", "date"]).size()
print(f"Daily counts distribution:\n{daily_counts.value_counts()}")

# Define window size and stride
window_size = 288  # 24 hours at 5-min intervals
stride = 144  # 50% overlap

# Create sliding windows
def create_windows(group):
    # Ensure group is sorted by time
    group = group.sort_values("time")

    # Check if the group has enough data points
    if len(group) < window_size:
        print(f"Skipping ID {group['ID'].iloc[0]} with only {len(group)} points.")
        return pd.DataFrame()  # Skip participants with too few data

    # Extract glucose values and timestamps
    values = group["glc"].values
    times = group["time"].values

    # Create sliding windows
    windows = []
    for i in range(0, len(values) - window_size + 1, stride):
        if i + window_size <= len(values):
            window = {
                "ID": group["ID"].iloc[0],
                "start_time": times[i],
                **{f"glc_{j}": values[i + j] for j in range(window_size)},
            }
            windows.append(window)

    return pd.DataFrame(windows)

# Apply windowing per participant
windows = df.groupby("ID").apply(create_windows).reset_index(drop=True)

# Validate window sizes
window_sizes = windows.apply(lambda x: len(x), axis=1)
print(f"Window sizes distribution:\n{window_sizes.value_counts()}")

# Save windows
windows.to_csv("data/processed/6_cgm_windows.csv", index=False)
print(f"Sliding windows created and saved!")
