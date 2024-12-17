import pandas as pd

input_file = "data/processed/6_cgm_windows.csv"
windows = pd.read_csv(input_file)

print(f"Expected window size: 288")
for i, window in enumerate(windows):
    if len(window) != 288:
        print(f"Mismatch at index {i}: window size = {len(window)}")
