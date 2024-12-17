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

# Process data in chunks
masked_data_chunks = []
mask_labels_chunks = []

print("Processing in chunks...")
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    print(f"Processing chunk with {len(chunk)} rows...")
    
    # Extract numeric columns
    glucose_columns = [col for col in chunk.columns if "glc_" in col]
    pos_enc_columns = [col for col in chunk.columns if "pe_" in col]

    glucose_data = chunk[glucose_columns].values  # Glucose data
    positional_encodings = chunk[pos_enc_columns].values  # Positional encodings

    # Mask glucose data
    masked_data = []
    mask_labels = []
    for window in glucose_data:
        masked_window, mask = mask_values(window, mask_prob)
        masked_data.append(masked_window)
        mask_labels.append(mask)
    
    # Combine masked glucose data with positional encodings
    masked_windows_with_pe = np.hstack([np.array(masked_data), positional_encodings])

    # Save masked data and labels for this chunk
    masked_data_chunks.append(masked_windows_with_pe)
    mask_labels_chunks.append(mask_labels)

# Save the combined results
print("Saving outputs...")
np.save(output_masked_file, np.vstack(masked_data_chunks))
np.save(output_labels_file, np.vstack(mask_labels_chunks))

print(f"Masked windows saved to: {output_masked_file}")
print(f"Mask labels saved to: {output_labels_file}")
