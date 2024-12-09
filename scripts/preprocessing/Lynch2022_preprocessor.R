# This is the script for processing Lynch2019 data into the common format.
# Author: Neo Kok
# Date: 10/24/2024
# Adjusted for project folder structure and compatibility

library(tidyverse)
library(haven)

# Define directories
raw_folder <- "data/raw/Lynch2022"  # Path to raw data folder
processed_folder <- "data/processed/cgm"  # Path for processed data output
dir.create(processed_folder, recursive = TRUE, showWarnings = FALSE)  # Ensure processed folder exists

# File paths
data_file <- file.path(raw_folder, "Data Tables in SAS", "iobp2devicecgm.sas7bdat")
demo_file <- file.path(raw_folder, "Data Tables in SAS", "iobp2diabscreening.sas7bdat")
age_file <- file.path(raw_folder, "Data Tables in SAS", "iobp2ptroster.sas7bdat")
output_file <- file.path(processed_folder, "lynch2022.csv")

# Read in data
data <- read_sas(data_file)
demo <- read_sas(demo_file)
age <- read_sas(age_file)

# Select only necessary variables
data <- data %>% select(PtID, DeviceDtTm, Value)
demo <- demo %>% select(PtID, InsModPump, Sex)
age <- age %>% select(PtID, AgeAsofEnrollDt)

# Merge variables on id
data <- left_join(data, demo, by = "PtID")
data <- left_join(data, age, by = "PtID")

# Rename and process columns
df_final <- data %>%
  select(
    id = PtID,
    time = DeviceDtTm,
    gl = Value,
    age = AgeAsofEnrollDt,
    sex = Sex,
    insulinModality = InsModPump
  ) %>%
  mutate(
    # Ensure correct time format
    time = as.POSIXct(time, format = "%m/%d/%Y %I:%M:%S %p"),
    # Set diabetes type to type 1
    type = as.numeric(1),
    # Set device type to Dexcom G6
    device = "Dexcom G6",
    # Set dataset type
    dataset = "lynch2022",
    # Set insulin modality: 0 for injections, 1 for pump
    insulinModality = as.numeric(ifelse(is.na(insulinModality), 0, 1))
  ) %>%
  # Remove NA times and glucose values
  filter(!is.na(time), !is.na(gl)) %>%
  group_by(id) %>%
  # Ensure that time is in order
  arrange(time) %>%
  # Generate pseudoID by adding 1000 to group IDs
  mutate(pseudoID = cur_group_id() + 1000) %>%
  # Ungroup the dataset
  ungroup() %>%
  # Reorder columns and select only relevant ones
  select(id = pseudoID, time, gl, age, sex, insulinModality, type, device, dataset)

# Save the processed dataset
write.csv(df_final, output_file, row.names = FALSE)
print(head(df_final))  # Check the first few rows of df_final
print(nrow(df_final))  # Check the number of rows in df_final
cat("Processing complete! Processed data saved to:", output_file, "\n")
