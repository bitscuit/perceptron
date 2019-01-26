import csv
import numpy as np
import pprint

trainingData = np.loadtxt('testSeeds.csv', delimiter=',')
normalized = trainingData / trainingData.max(axis=0)

# def importCSV(filename):
#     with open (filename, newline='') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         for row in reader:
#             trainingData.append([float(r) for r in row])

# importCSV('testSeeds.csv')
pprint.pprint(normalized)