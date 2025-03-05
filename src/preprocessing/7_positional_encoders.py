import numpy as np
import pandas as pd

# File paths
input_file = "data/processed/6_cgm_windows.csv"
output_file = "data/processed/7_cgm_windows_with_pe_test.csv"

# Parameters
window_size = 288
embed_dim = 32
chunk_size = 1000  # Keep it manageable

# Positional encoding function (without flattening)
def positional_encoding(window_size, embed_dim):
    positions = np.arange(window_size).reshape(-1, 1)
    div_terms = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    sinusoidals = np.zeros((window_size, embed_dim))
    sinusoidals[:, 0::2] = np.sin(positions * div_terms)  # Sin for even indices
    sinusoidals[:, 1::2] = np.cos(positions * div_terms)  # Cos for odd indices
    return sinusoidals

# Generate PE **without flattening**
pe = positional_encoding(window_size, embed_dim)  # Shape: (288, 32)

# **New column naming convention**
glucose_colnames = [f"glc_{t}" for t in range(window_size)]
pe_colnames = [f"pe_{t}_{d}" for t in range(window_size) for d in range(embed_dim)]

# Process in chunks and save incrementally
header_written = False
last_user = None  # Track previous user

for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
    print(f"Processing chunk {chunk_idx + 1} with {len(chunk)} rows...")

    # **Extract user ID before dropping it**
    user_ids = chunk["ID"].values  

    # **Drop ID and start_time before further processing**
    chunk = chunk.drop(columns=["ID", "start_time"])  

    # Extract glucose columns
    glucose_columns = [col for col in chunk.columns if col.startswith("glc_")]
    glucose_data = chunk[glucose_columns].values  # Shape: (chunk_size, 288)

    # **Repeat PE for all rows in this chunk**
    pe_repeated = np.tile(pe, (len(glucose_data), 1, 1))  # Shape: (chunk_size, 288, 32)
    pe_reshaped = pe_repeated.reshape(len(glucose_data), -1)  # Flatten to (chunk_size, 288*32)

    # **Insert padding row when user changes**
    cleaned_glucose_data = []
    cleaned_pe_data = []  # New list for PE padding

    for i in range(len(glucose_data)):
        user_id = user_ids[i]

        # If user ID changes, insert a padding row
        if last_user is not None and user_id != last_user:
            padding_glucose = np.full((1, 288), -1)  # ✅ Glucose padding
            padding_pe = np.full((1, 288 * 32), -1)  # ✅ PE padding

            cleaned_glucose_data.append(padding_glucose)
            cleaned_pe_data.append(padding_pe)

            print(f"Inserted padding row between {last_user} -> {user_id}")

        # Append normal data
        cleaned_glucose_data.append(glucose_data[i])
        cleaned_pe_data.append(pe_reshaped[i])  # ✅ Use pre-repeated PE

        last_user = user_id  # Update last seen user

    # Convert lists to NumPy arrays
    cleaned_glucose_data = np.vstack(cleaned_glucose_data)  # ✅ Ensures rectangular shape
    cleaned_pe_data = np.vstack(cleaned_pe_data)  # ✅ Ensures rectangular shape

    # Convert to DataFrame
    glucose_df = pd.DataFrame(cleaned_glucose_data, columns=glucose_colnames)
    pe_df = pd.DataFrame(cleaned_pe_data, columns=pe_colnames)  # ✅ Use cleaned PE data

    # Combine glucose + PE
    final_df = pd.concat([glucose_df, pe_df], axis=1)

    print("Final shape before saving:", final_df.shape)  # ✅ Debugging check

    # **Write to CSV immediately (incremental saving)**
    final_df.to_csv(output_file, mode="a", index=False, header=not header_written)
    header_written = True  # Ensures header is only written once

print(f"Windows with positional encodings saved to: {output_file}")
