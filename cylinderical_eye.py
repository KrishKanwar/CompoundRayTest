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
        z_range = 21
        acceptance = 10.0  # example static value

        points_num = 100

        for j in range(z_range):
            # Write 16 rows of data
            for i in range(points_num):
                angle = (360 / points_num) * i
                x = 1 * np.cos(np.radians(angle))
                y = 1 * np.sin(np.radians(angle))
                # z = (1/7) * (j - ((z_range - 1) / 2))
                z = 100000 * (j - ((z_range - 1) / 2)) / ((z_range - 1) / 2)
                vx = np.cos(np.radians(angle))
                vy = np.sin(np.radians(angle))
                # vz = (j - (z_range / 2)) / (z_range / 2)  # vz is static
                vz = 0

                # Write the row
                writer.writerow([x, y, z, vx, vy, vz, acceptance])

    print(f"CSV file '{file_path}' created successfully with data.")


# Example usage:
create_csv_with_data("cylinderical_eye_data.csv")
