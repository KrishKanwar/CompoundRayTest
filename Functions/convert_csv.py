import csv
import math
import sys

import argparse

argv = sys.argv
argv = argv[1:]

usage_text = "Usage:" + "  convertcsv.py" + " [options]"
parser = argparse.ArgumentParser(description=usage_text)
parser.add_argument(
    "-i", "--inputdir", dest="input", type=str, default="", help="input file path"
)
parser.add_argument(
    "-o", "--output", dest="output", type=str, default=None, help="output file path"
)

if not argv:
    parser.print_help()
    exit()

args = parser.parse_args(argv)

input_path = args.input
output_path = args.output

with open(input_path, "r") as csvinput:
    with open(output_path, "w") as csvoutput:
        writer = csv.writer(csvoutput, delimiter=" ", lineterminator="\n")
        reader = csv.reader(csvinput)

        # skip labels
        next(reader)

        for row in reader:
            line = []
            # convert the coordinate system ((x, y, z) -> (y, z, x)) and the scale (um -> mm)
            line.append(str(float(row[1]) / 1000.0))
            line.append(str(float(row[2]) / 1000.0))
            line.append(str(float(row[0]) / 1000.0))
            # convert the vector of the eye direction ((x, y, z) -> (y, z, x))
            line.append(row[4])
            line.append(row[5])
            line.append(row[3])
            # convert the acceptance angle (degree -> radian)
            line.append(str(float(row[6]) * math.pi / 180.0))
            # set the offset to zero
            line.append("0.0")
            writer.writerow(line)
