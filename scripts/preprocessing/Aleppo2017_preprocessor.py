import os
import datetime

# Paths
dataset = "data/raw/Aleppo2017"
file = os.path.join(dataset, "DataTables", "HDeviceCGM.txt")
output_dir = "data/processed/Aleppo2017"
os.makedirs(output_dir, exist_ok=True)  # Ensure the processed directory exists
newfile = os.path.join(output_dir, "cgm_processed.csv")

# Base date for the study
basedate = datetime.date(2015, 5, 22)

# Process the CGM file
try:
    with open(file, "r") as file:  # Open the input file
        with open(newfile, "w") as export:  # Open the output file
            isheader = True  # Flag to track the header line
            
            for line in file:
                # Skip the header and write a new one
                if isheader:
                    isheader = False
                    export.write("\"id\",\"time\",\"gl\"\n")
                    continue
                
                # Split the line by the '|' delimiter
                line = line.strip().split('|')
                
                # Parse the date and time
                day = datetime.timedelta(days=int(line[4]))
                thisdate = basedate + day
                thistime = datetime.datetime.strptime(line[5], "%H:%M:%S").time()
                thedatetime = datetime.datetime.combine(thisdate, thistime)

                # Extract glucose value and format the output
                glucose_value = line[9][0:3]  # First three characters of the glucose field
                export.write(f"{line[2]},{thedatetime},{glucose_value}\n")

    print(f"Processing complete! Processed data saved to: {newfile}")

except FileNotFoundError as e:
    print(f"Error: File not found. Check the path: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
