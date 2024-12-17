import numpy as np
import pandas as pd

# Load the windowed data
chunk_size = 288 * 34  # Number of rows to process in one chunk
input_file = "data/processed/6_cgm_windows.csv"
output_file = "data/processed/7_cgm_windows_with_pe.csv"

# Define positional encoding function
def positional_encoding(window_size, embed_dim):
    positions = np.arange(window_size).reshape(-1, 1)
    div_terms = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    sinusoidals = np.zeros((window_size, embed_dim))
    sinusoidals[:, 0::2] = np.sin(positions * div_terms)  # Apply sin to even indices
    sinusoidals[:, 1::2] = np.cos(positions * div_terms)  # Apply cos to odd indices
    return sinusoidals

# Embedding parameters
embed_dim = 64
window_size = 288  # Expected number of time steps per window

# Process in chunks
header_written = False
for chunk in pd.read_csv(input_file, chunksize=chunk_size):
    print(f"Processing chunk with {len(chunk)} rows...")

    # Generate positional encodings for the window size
    pos_enc = positional_encoding(window_size, embed_dim)

    # Add positional encodings to the chunk
    chunk = chunk.reset_index(drop=True)  # Ensure chunk is clean
    pe_columns = pd.DataFrame(
        pos_enc,
        columns=[f"pe_{i}" for i in range(embed_dim)]
    )
    print(pe_columns.head())
    # Repeat positional encodings for each row in the chunk
    pe_repeated = pd.concat([pe_columns] * len(chunk), ignore_index=True)

    # Combine the chunk with the positional encodings
    chunk_with_pe = pd.concat([chunk.reset_index(drop=True), pe_repeated], axis=1)

    # Save to file
    chunk_with_pe.to_csv(output_file, mode="a", index=False, header=not header_written)
    header_written = True

print(f"Windows with positional encodings saved to: {output_file}")
