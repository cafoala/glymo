import numpy as np
import pandas as pd
import os

# Parameters
input_file = "data/processed/7_cgm_windows_with_pe.csv"
output_masked_file = "data/processed/masked_windows.npy"
output_labels_file = "data/processed/mask_labels.npy"

chunk_size = 288 * 34  # Number of rows to process in one chunk
mask_prob = 0.2  # Probability of masking

# Ensure output folder exists
os.makedirs(os.path.dirname(output_masked_file), exist_ok=True)

# Masking function
def mask_values(window, mask_prob=0.2):
    mask = np.random.rand(window.shape[0]) < mask_prob  # Create mask
    masked_window = window.copy()
    masked_window[mask] = -1  # Replace masked positions with -1
    return masked_window, mask

# Process data in chunks and save incrementally
masked_data_file = open(output_masked_file, "wb")
mask_labels_file = open(output_labels_file, "wb")

print("Processing in chunks...")
for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
    print(f"Processing chunk {chunk_idx + 1} with {len(chunk)} rows...")
    
    # Extract numeric columns
    glucose_columns = [col for col in chunk.columns if "glc_" in col]
    pos_enc_columns = [col for col in chunk.columns if "pe_" in col]

    glucose_data = chunk[glucose_columns].values  # Glucose data
    positional_encodings = chunk[pos_enc_columns].values  # Positional encodings

    # Mask glucose data and combine with positional encodings
    masked_data_chunk = []
    mask_labels_chunk = []
    
    for window in glucose_data:
        masked_window, mask = mask_values(window, mask_prob)
        masked_data_chunk.append(np.hstack([masked_window, positional_encodings[0]]))  # Combine
        mask_labels_chunk.append(mask)  # Mask labels

    # Save directly to files
    np.save(masked_data_file, np.array(masked_data_chunk))
    np.save(mask_labels_file, np.array(mask_labels_chunk))

masked_data_file.close()
mask_labels_file.close()

print(f"Masked windows saved to: {output_masked_file}")
print(f"Mask labels saved to: {output_labels_file}")
