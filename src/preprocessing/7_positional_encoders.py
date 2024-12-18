import numpy as np
import pandas as pd

# File paths
input_file = "data/processed/6_cgm_windows.csv"
output_file = "data/processed/7_cgm_windows_with_pe_test.csv"

# Parameters
window_size = 288
embed_dim = 32
chunk_size = 10000  # Rows to process in one chunk

# Positional encoding function
def positional_encoding(window_size, embed_dim):
    positions = np.arange(window_size).reshape(-1, 1)
    div_terms = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    sinusoidals = np.zeros((window_size, embed_dim))
    sinusoidals[:, 0::2] = np.sin(positions * div_terms)  # Apply sin to even indices
    sinusoidals[:, 1::2] = np.cos(positions * div_terms)  # Apply cos to odd indices
    return sinusoidals

# Generate PE for one window
pe = positional_encoding(window_size, embed_dim)  # Shape: (288, 64)
pe_flattened = pe.flatten()  # Shape: (288 * 64,)

# Process in chunks
header_written = False
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    print(f"Processing chunk with {len(chunk)} rows...")

    # Extract glucose columns only (exclude ID and start_time)
    glucose_columns = [col for col in chunk.columns if col.startswith("glc_")]
    glucose_data = chunk[glucose_columns].values  # Shape: (len(chunk), 288)

    # Repeat PE for all rows in the chunk
    pe_repeated = np.tile(pe_flattened, (len(chunk), 1))  # Shape: (len(chunk), 18,432)

    # Combine glucose data and PE
    combined = np.hstack([glucose_data, pe_repeated])  # Shape: (len(chunk), 18,720)

    # Combine with metadata (`ID` and `start_time`)
    metadata = chunk[["ID", "start_time"]].reset_index(drop=True)  # Keep metadata as is
    combined_df = pd.concat(
        [metadata, pd.DataFrame(combined)], axis=1
    )  # Metadata + Glucose + PE

    # Save combined data
    combined_df.to_csv(output_file, mode="a", index=False, header=not header_written)
    print(combined_df.head(2))
    header_written = True

print(f"Windows with positional encodings saved to: {output_file}")
