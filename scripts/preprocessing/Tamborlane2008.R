# This is the script for processing the Tamborlane2008 data into the common format.
# Author: David Buchanan
# Date: January 31st, 2020, edited June 13th, 2020 by Elizabeth Chun
# Updated for compatibility and directory handling

library(tidyverse)

# Define paths
raw_folder <- "data/raw/Tamborlane2008"  # Raw data folder
processed_file <- "data/processed/cgm/tamborlane_processed_cgm.csv"  # Processed output file

# Ensure the processed directory exists
dir.create(dirname(processed_file), recursive = TRUE, showWarnings = FALSE)

# Path to the CGM files
cgm_folder <- file.path(raw_folder, "DataTables")

# List all files matching the CGM pattern
files <- list.files(cgm_folder, pattern = "RTCGM", full.names = TRUE)

# Loop through the files
for (i in seq_along(files)) {
  filename <- files[i]  # Get the file name
  print(paste("Processing file:", filename))
  
  # Read the CSV file
  curr <- read.csv(filename, stringsAsFactors = FALSE)
  
  # Remove unnecessary columns
  if ("RecID" %in% colnames(curr)) {
    curr <- curr %>% select(-RecID)
  }
  
  # Rename columns to standard column names
  colnames(curr) <- c("id", "time", "gl")
  
  # Normalize and parse the `time` column
  curr$time <- strptime(curr$time, format = "%Y-%m-%d %H:%M:%S", tz = "") %>%
    as.POSIXct()

  # Ensure glucose values are numeric
  curr$gl <- as.numeric(curr$gl)
  
  # Remove rows with missing time or glucose values
  curr <- curr %>% filter(!is.na(time) & !is.na(gl))
  
  # Append the cleaned data to the processed file
  write.table(curr, file = processed_file, sep = ",", row.names = FALSE,
              col.names = !file.exists(processed_file), append = TRUE)
}

print(paste("Processing complete! Processed data saved to:", processed_file))
