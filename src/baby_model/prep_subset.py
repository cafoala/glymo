import numpy as np

# Paths
input_file = "data/processed/masked_windows.npy"
label_file = "data/processed/mask_labels.npy"
output_subset_file = "data/processed/masked_windows_subset.npy"
output_labels_subset_file = "data/processed/mask_labels_subset.npy"

# Load the full masked data
masked_windows = np.load(input_file)
mask_labels = np.load(label_file)

# Select a subset
subset_size = 1000  # Adjust based on your needs
masked_windows_subset = masked_windows[:subset_size]
mask_labels_subset = mask_labels[:subset_size]

# Save the subset
np.save(output_subset_file, masked_windows_subset)
np.save(output_labels_subset_file, mask_labels_subset)

print(f"Subset saved to {output_subset_file} and {output_labels_subset_file}")
