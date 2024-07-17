import csv
import math
import numpy as np


def create_csv_with_data(file_path):
    # Define the column headers
    headers = ["x", "y", "z", "vx", "vy", "vz", "acceptance"]

    # Open the file in write mode
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(headers)

        # Define a static value for z and acceptance
        z = 0.0  # example static value
        acceptance = 0.0  # example static value

        points_num = 16

        for j in range(5):
            # Write 16 rows of data
            for i in range(points_num):
                angle = (360/points_num)*i
                x = np.cos(np.radians(angle))
                y = np.sin(np.radians(angle))
                vx = x - 0
                vy = y - 0
                vz = 0  # vz is static

                # Write the row
                writer.writerow([x, y, z, vx, vy, vz, acceptance])

    print(f"CSV file '{file_path}' created successfully with data.")


# Example usage:
create_csv_with_data("output_with_data.csv")
