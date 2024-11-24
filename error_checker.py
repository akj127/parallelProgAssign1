import csv
import math
import argparse

def calculate_average_percentage_error(file1, file2):
    values1 = []
    values2 = []

    # Read file1
    with open(file1, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                value = float(row[0])
                if not math.isinf(value):
                    values1.append(value)
            except ValueError:
                print(f"Error: Non-numeric value in file1: {row[0]}")

    # Read file2
    with open(file2, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                value = float(row[0])
                if not math.isinf(value):
                    values2.append(value)
            except ValueError:
                print(f"Error: Non-numeric value in file2: {row[0]}")

    if len(values1) != len(values2):
        raise ValueError("Files have different lengths.")

    total_error = 0
    valid_values = 0
    under_5_percent_count = 0  # To track how many percentage errors are under 5%

    # Calculate errors
    for i in range(len(values1)):
        if math.isinf(values1[i]) or math.isinf(values2[i]):
            print(f"Warning: Infinite value encountered at index {i}")
            continue

        absolute_difference = abs(values1[i] - values2[i])
        if values1[i] != 0:
            relative_difference = absolute_difference / values1[i]
            percentage_difference = relative_difference * 100
            total_error += percentage_difference
            valid_values += 1

            # Check if the percentage difference is less than 5%
            if percentage_difference < 5:
                under_5_percent_count += 1

    if valid_values == 0:
        return "No valid values for calculation"

    average_percentage_error = total_error / valid_values

    return average_percentage_error, under_5_percent_count / valid_values

# Main function to handle command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate average percentage error between two CSV files.')
    parser.add_argument('file1', type=str, help='The first CSV file (oracle values)')
    parser.add_argument('file2', type=str, help='The second CSV file (computed values)')
    
    args = parser.parse_args()
    
    average_error, under_5_percent_count = calculate_average_percentage_error(args.file1, args.file2)
    print("Average percentage error:", average_error)
    print("Proportion of percentage errors under 5%:", under_5_percent_count)
