import pandas as pd
import logging

# Configure logging
logging.basicConfig(filename="process_data_log.txt", level=logging.INFO,
                    format="%(asctime)s %(message)s")

# Part B: Import and Format Data
# Load raw Florida data
logging.info("Loading raw data from airportdataraw.csv")
df = pd.read_csv("airportdataraw.csv")

# Part C: Initial Cleaning Step (Filter for MIA)
# Filter for MIA departures to focus on the chosen airport
df = df[df["ORIGIN"] == "MIA"]

# Part B: Continue Formatting
# Select and rename columns to match the expected format for poly_regressor_Python_1.0.0.py
columns_needed = {
    "YEAR": "YEAR",
    "MONTH": "MONTH",
    "DAY_OF_MONTH": "DAY",
    "DAY_OF_WEEK": "DAY_OF_WEEK",
    "ORIGIN": "ORG_AIRPORT",
    "DEST": "DEST_AIRPORT",
    "CRS_DEP_TIME": "SCHEDULED_DEPARTURE",
    "DEP_TIME": "DEPARTURE_TIME",
    "DEP_DELAY": "DEPARTURE_DELAY",
    "CRS_ARR_TIME": "SCHEDULED_ARRIVAL",
    "ARR_TIME": "ARRIVAL_TIME",
    "ARR_DELAY": "ARRIVAL_DELAY"
}
df = df[list(columns_needed.keys())].rename(columns=columns_needed)

# Part B: Ensure integer types and handle missing values
# Convert numeric columns to integers and coerce invalid values to NaN
for col in ["YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_DEPARTURE", 
            "DEPARTURE_TIME", "DEPARTURE_DELAY", "SCHEDULED_ARRIVAL", 
            "ARRIVAL_TIME", "ARRIVAL_DELAY"]:
    df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

# Part B: Drop rows with NaN in critical columns
# Ensure no missing values in required columns to avoid errors in the model
df = df.dropna(subset=columns_needed.values())

# Part C: Additional Cleaning Steps
# Cleaning Step 1: Remove delays > 60 minutes
# This aligns with poly_regressor_Python_1.0.0.py's logic to exclude extreme delays,
# prevents outliers from skewing the polynomial regression, and focuses on typical delays
df = df[df["DEPARTURE_DELAY"] <= 60]
logging.info("Removed delays > 60 minutes, new shape: %s", df.shape)

# Cleaning Step 2: Replace negative delays with 0
# Negative delays (early departures) are set to 0 because the model focuses on predicting delays,
# not earliness, and negative values could skew performance metrics like MSE
df["DEPARTURE_DELAY"] = df["DEPARTURE_DELAY"].apply(lambda x: max(x, 0))
logging.info("Replaced negative delays with 0")

# Part C: Save cleaned data as final output
df.to_csv("cleaned_data.csv", index=False)
logging.info("Final cleaned data saved as cleaned_data.csv with shape: %s", df.shape)
print("Final cleaned data:\n", df.head())