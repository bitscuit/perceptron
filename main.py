import csv
import numpy as np
import pprint

trainingData = np.loadtxt('testSeeds.csv', delimiter=',') # read csv file
normalized = trainingData / trainingData.max(axis=0) # normalize data

def yeetWheat(bias, input, weights):
    # add bias for neuron to determine it's own 
    output = bias * weights[0]
    output += sum([x * w for x, w in zip(input[:-1], weights[1:])])
    print(output)
    return 1 if output > 0 else 0

def perceptron():
    bias = 1
    input = [1, 2, 3, 4, 5, 6, 7, 8]
    weights1 = [-0.1, 1, 0.1, 1, 0.1, -1, 0.1, -1]
    weights2 = [-1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1]

    y1 = yeetWheat(bias, input, weights1)
    y2 = yeetWheat(bias, input, weights2)
    return str(y1) + str(y2)

# pprint.pprint(trainingData)
# pprint.pprint(normalized)

print(perceptron())