# Author: Samuel Tan, Neo Kok
# Date: 10/24/2024

library(tidyverse)
library(haven)
library(lubridate)

# Define directories
raw_folder <- "data/raw/OMalley2021"
processed_folder <- "data/processed/OMalley2021"
dir.create(processed_folder, recursive = TRUE, showWarnings = FALSE)  # Ensure processed folder exists

# File paths
cgm_file <- file.path(raw_folder, "Data Files", "cgm.txt")
screening_file <- file.path(raw_folder, "Data Files", "DiabScreening_a.txt")

# Read in necessary data
cgmData <- read.table(cgm_file, sep = "|", header = TRUE)
screeningData <- read.table(screening_file,
                            sep = "|", header = TRUE, fill = TRUE, 
                            fileEncoding = "UTF-16LE", stringsAsFactors = FALSE, na.strings = "")

# Merge demographic data with CGM data
merged_data <- cgmData %>%
  left_join(screeningData, by = "PtID")

# Add additional variables: specify the dataset, subject type, device used, and placeholder values
final_data <- merged_data %>%
  mutate(
    id = PtID,
    time = as.POSIXct(dmy_hms(DataDtTm), format = "%Y-%m-%d %H:%M:%S"),
    gl = as.numeric(CGM),
    age = AgeAtEnrollment,
    sex = Gender,
    insulinModality = as.numeric(1),
    type = as.numeric(1),
    device = "Dexcom G6",
    dataset = "o_malley2021"
  ) %>%
  # Remove NA times and gl values
  filter(!is.na(time), !is.na(gl)) %>%
  group_by(id) %>%
  # Ensure that time is in order
  arrange(time) %>%
  # Generate unique pseudo IDs for each participant by adding 5000 to group IDs
  mutate(pseudoID = cur_group_id() + 5000) %>%
  # Ungroup the dataset after creating pseudoID
  ungroup() %>%
  # Select necessary variables
  select(id = pseudoID, time, gl, age, sex, insulinModality, type, device, dataset)

# Save the processed dataset to the processed folder
output_file <- file.path(processed_folder, "o_malley2021.csv")
final_data %>% write.csv(output_file, row.names = FALSE)

cat("Processing complete! Processed data saved to:", output_file, "\n")
