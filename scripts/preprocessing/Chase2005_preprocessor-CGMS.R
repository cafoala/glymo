# This is the script for processing children200 CGMS data into the common format.
# Author: Elizabeth Chun
# Date: September 23rd, 2020

# Define paths
dataset <- "data/raw/Chase2005"  # Path to the dataset folder
file.path <- file.path(dataset, "DataTables", "tblCDataCGMS.csv")  # Path to the CGM file
output_dir <- "data/processed/Chase2005"  # Path for processed data output
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)  # Create output directory if it doesn't exist
output_file <- file.path(output_dir, "chase_cgm_processed.csv")  # Path for the processed file

# Read the raw data
curr <- read.csv(file.path, header = TRUE, stringsAsFactors = FALSE)
old <- curr

# Detect AM/PM in the ReadingTm column
indexAM <- grep("AM", curr$ReadingTm)
indexPM <- grep("PM", curr$ReadingTm)
timeInfo <- paste(as.Date(curr$ReadingDt), curr$ReadingTm)

# Initialize time vector
time <- rep(as.POSIXct(NA), length(timeInfo))

# First, attempt AM/PM conversion
tz <- "EST"  # Timezone
time[indexAM] <- as.POSIXct(timeInfo[indexAM], format = "%Y-%m-%d %I:%M %p", tz = tz)
time[indexPM] <- as.POSIXct(timeInfo[indexPM], format = "%Y-%m-%d %I:%M %p", tz = tz)

# Replace remaining NA values with 24-hour format
newtime <- as.POSIXct(timeInfo, format = "%Y-%m-%d %H:%M", tz = tz)
time[is.na(time)] <- newtime[is.na(time)]

# Combine date and time into a single column
curr$time <- time

# Reorder and select only id, time, gl columns
curr <- curr[, c(2, 7, 6)]

# Rename columns to the standard format
colnames(curr) <- c("id", "time", "gl")

# Convert glucose to numeric
curr$gl <- as.numeric(curr$gl)

# Save the processed data
write.table(curr, file = output_file, row.names = FALSE,
            col.names = !file.exists(output_file), sep = ",")

cat("Processing complete! Processed data saved to:", output_file, "\n")
