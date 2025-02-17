import os
import pandas as pd
import time
import sys

# Initialize CSV file with header to store results
def initialize_csv(file_path):
    # Columns: Turn (Episode), Mean, and factors
    columns = ["countNum","Step", "Mean", "Factor_1", "Factor_2", "Factor_3"]
    pd.DataFrame(columns=columns).to_csv(file_path, index=False)

# Function to log results to the CSV file
def log_to_csv(file_path, countNum,step, stateMean):
    # Prepare data for logging
    data = {"countNum": countNum,"Step": step, "Mean": stateMean[-1]}
    for i, value in enumerate(stateMean[:-1]):
        data[f"Factor_{i+1}"] = value

    pd.DataFrame([data]).to_csv(file_path, mode='a', header=False, index=False)

def log_factors(file_path, countNum, step, factor_1, factor_2, factor_3, mean=None):
    new_data = pd.DataFrame([[countNum, step, mean, factor_1, factor_2, factor_3]],
                            columns=["countNum", "Step", "Mean", "Factor_1", "Factor_2", "Factor_3"])
    new_data.to_csv(file_path, mode='a', header=False, index=False)
    print("to csv done")
    #full_data = pd.read_csv(file_path)
    #indexNum = len(full_data)
    #print("index",indexNum)
    return mean #,indexNum

def get_mean_for_step(file_path,countNum):
    max_attempts = 30  # Maximum number of times to wait
    attempts = 0       # Counter for attempts

    while attempts < max_attempts:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if the DataFrame has at least one row
        if not df.empty and 'Mean' in df.columns:
            last_value = df.iloc[countNum-1]['Mean']

            # Check if the value is numeric
            if isinstance(last_value, (int, float)) and not pd.isna(last_value):
                mean_value = last_value
                print(f"Mean value found: {mean_value}")
                return mean_value
            else:
                print(f"Attempt {attempts + 1}: Waiting for a numeric value in the last row of 'mean' column...")

        else:
            print("The DataFrame is empty or 'mean' column is missing.")

        # Wait for 2 seconds before re-checking
        time.sleep(5)
        attempts += 1

    # If maximum attempts are reached without finding a numeric value
    print("Reached maximum wait time without finding a numeric value in the last row of 'mean' column.")
    sys.exit(1)

