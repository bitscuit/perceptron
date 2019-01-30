import numpy as np
import pprint



ITERATIONS = 200
LEARNING_RATE = 0.01
BIAS = 1

count = 0


# reads the dataset from CSV file and normalizes it
def getData(filename):
    data = np.loadtxt(filename, delimiter=',') # load csv file into numpy array
    # normalize data (ignore last column because that is desired output)
    normalized = data[:, :-1] / data[:, :-1].max(axis=0)
    # put normalized data back into array with desired output column
    data[:, :-1] = normalized
    return data


# output function of neuron which calculates the weighted sum of inputs
# the output is 1 if sum is greater than 0, 0 otherwise
def activate(input, weights):
    # add bias for neuron to determine its own threshold
    output = BIAS * weights[0]
    # list comprehension to perform element wise multiplication of the input
    # and weights lists
    output += sum([x * w for x, w in zip(input, weights[1:])])
    return 1 if output > 0 else 0


# implementation of a 2 neuron perceptron where weights1 is the set of input
# weights feeding into neuron 1 and weights2 is the set of input weights feeding
# into neuron 2
# the outputs are concatenated and treated like a binary string
def perceptron(input, weights1, weights2):
    y1 = activate(input, weights1)
    y2 = activate(input, weights2)
    return str(y1) + str(y2)


# helper function to update the weights of perceptron while training
def updateWeights(x, w, y, d):
    weights = []
    # update weight of BIAS input, resulting in perceptron determining its own
    # threshold
    weights.append(w[0] + ((d - y) * LEARNING_RATE * BIAS))
    # update weight, w, for each input, x
    for i in range(len(x)):
        weights.append(w[i+1] + ((d - y) * LEARNING_RATE * x[i]))
    return weights


def trainPerceptron(input, weights1, weights2):
    global count
    for i in range(ITERATIONS):
        for row in input:
            y = perceptron(row[:-1], weights1, weights2)
            # print('y is: {}, and d is: {}'.format(int(y,2), row[len(row)-1]))
            d = row[len(row)-1]
            test = format(int(d), '02b')
            if int(test[0]) - int(y[0]) != 0:
                weights1 = updateWeights(row[:-1], weights1, int(y[0]), int(test[0]))
            if int(test[1]) - int(y[1]) != 0:
                weights2 = updateWeights(row[:-1], weights2, int(y[1]), int(test[1]))
            if int(y, 2) - int(row[len(row)-1]) == 0:
                count += 1
            # print(y)
        print('This many were right: {}'.format(count))     
        count = 0
    print(weights1)
    print(weights2)
    return (weights1, weights2)


def testPerceptron(input, weights1, weights2):
    global count
    count = 0
    for row in input:
        y = perceptron(row[:-1], weights1, weights2)
        d = row[len(row)-1]
        # print(y)
        if int(y, 2) - int(d) == 0:
            count +=1
    print('This many were right: {}'.format(count))         



def main():
    weights1 = [-0.1, 1, 0.1, 1, 0.1, -1, 0.1, -1]
    weights2 = [-1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1]
    # first weight is for bias
    # weights1 = [-0.3600000000000002, 1.0124787535410458, 0.008660869565218606, 0.3704475661548351, -0.14466367041199163, -1.1815497148523761, 0.125947634237115, 0.47409770992366457]
    # weights2 = [-0.3399999999999994, -1.1172521246458798, 0.544226086956539, 0.7400718719372663, 0.7303610486891207, -0.307255145053313, -0.1705348219032502, -0.35072519083968534]

    trainingData = getData('trainSeeds.csv')
    np.random.shuffle(trainingData)

    # train perceptron
    weights1, weights2 = trainPerceptron(trainingData, weights1, weights2)

    testData = getData('testSeeds.csv')
    np.random.shuffle(testData)

    testPerceptron(testData, weights1, weights2)


if __name__ == '__main__':
    main()
