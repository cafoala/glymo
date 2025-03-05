import numpy as np
import pandas as pd
import os

# Parameters
input_file = "data/processed/7_cgm_windows_with_pe_test.csv"
output_masked_file = "data/processed/masked_windows.npy"
output_labels_file = "data/processed/mask_labels.npy"

chunk_size = 10000  # Number of rows to process in one chunk
mask_prob = 0.2  # Probability of masking

# Ensure output folder exists
os.makedirs(os.path.dirname(output_masked_file), exist_ok=True)

# Masking function
def mask_values(window, mask_prob=0.2):
    mask = np.random.rand(*window.shape) < mask_prob  # Create mask (match window shape)
    masked_window = window.copy()
    masked_window[mask] = -1  # Replace masked positions with -1
    return masked_window, mask

# Storage for final dataset
all_masked_data = []
all_mask_labels = []

print("Processing in chunks...")
for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
    print(f"Processing chunk {chunk_idx + 1} with {len(chunk)} rows...")

    # Extract glucose and positional encoding columns
    glucose_columns = [col for col in chunk.columns if "glc_" in col]
    pos_enc_columns = [col for col in chunk.columns if "pe_" in col]

    glucose_data = chunk[glucose_columns].values  # Extract glucose values
    positional_encodings = chunk[pos_enc_columns].values  # Extract positional encodings

    print(f"Glucose data shape: {glucose_data.shape}")
    print(f"Positional encoding shape: {positional_encodings.shape}")

    # Ensure shape consistency
    if glucose_data.shape[0] != positional_encodings.shape[0]:
        raise ValueError(
            f"Mismatch: glucose_data has {glucose_data.shape[0]} rows, "
            f"positional_encodings has {positional_encodings.shape[0]} rows."
        )

    # Mask glucose data and combine with positional encodings
    masked_data_chunk = []
    mask_labels_chunk = []

    for i in range(len(glucose_data)):
        masked_window, mask = mask_values(glucose_data[i], mask_prob)
        combined_window = np.hstack([masked_window, positional_encodings[i]])  # Combine masked glucose + PE
        masked_data_chunk.append(combined_window)
        mask_labels_chunk.append(mask)

    # Convert to NumPy arrays and ensure shape is correct
    masked_data_chunk = np.array(masked_data_chunk, dtype=np.float32)  # Ensure consistent dtype
    mask_labels_chunk = np.array(mask_labels_chunk, dtype=np.float32)  # Ensure consistent dtype

    all_masked_data.append(masked_data_chunk)
    all_mask_labels.append(mask_labels_chunk)

# Concatenate all chunks and ensure the final shape is correct
all_masked_data = np.concatenate(all_masked_data, axis=0)[:, :9216]  # **Force correct number of features**
all_mask_labels = np.concatenate(all_mask_labels, axis=0)

np.save(output_masked_file, all_masked_data)
np.save(output_labels_file, all_mask_labels)

print(f"Masked windows saved to: {output_masked_file}, Shape: {all_masked_data.shape}")
print(f"Mask labels saved to: {output_labels_file}, Shape: {all_mask_labels.shape}")
