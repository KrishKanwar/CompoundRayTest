import csv
import math

with open('/home/kawaset@hhmi.org/compound-ray/data/eyes/lens_opticAxis_acceptance.csv','r') as csvinput:
    with open('/home/kawaset@hhmi.org/compound-ray/data/eyes/lens_opticAxis_acceptance.eye', 'w') as csvoutput:
        writer = csv.writer(csvoutput, delimiter=' ', lineterminator='\n')
        reader = csv.reader(csvinput)

        next(reader)

        for row in reader:
            all = []
            all.append( str(float(row[1]) / 1000.0) )
            all.append( str(float(row[2]) / 1000.0) )
            all.append( str(float(row[0]) / 1000.0) )
            all.append(row[4])
            all.append(row[5])
            all.append(row[3])
            all.append( str(float(row[6]) * math.pi / 180.0) )
            all.append('0.0')
            writer.writerow(all)