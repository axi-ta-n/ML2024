import math
import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))  # Return the sigmoid of inX

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)  # Get the number of samples (m) and number of features (n)
    weights = np.ones(n)  # Initialize the weights vector with ones

    for j in range(numIter):  # Iterate over the number of iterations
        dataIndex = list(range(m))  # Create a list of indices for the data points
        for i in range(m):  # Iterate over each sample in the dataset
            alpha = 4 / (1.0 + j + i) + 0.01  # Set the learning rate, which decreases with each iteration
            randIndex = int(random.uniform(0, len(dataIndex)))  # Select a random index from the remaining data points
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # Compute the hypothesis (predicted value) for the randomly selected sample
            error = classLabels[randIndex] - h  # Calculate the error (difference between actual and predicted value)
            weights = weights + alpha * error * dataMatrix[randIndex]  # Update the weights using the stochastic gradient ascent rule
            del(dataIndex[randIndex])  # Remove the selected index from the list of indices to avoid re-selection in the same iteration

    return weights  # Return the optimized weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))  # Compute the probability using the sigmoid function
    if prob > 0.5:  # If the probability is greater than 0.5
        return 1.0  # Classify as 1
    else:  # If the probability is 0.5 or less
        return 0.0  # Classify as 0

def colicTest():
    frTrain = open('D:\\ML Internship\\logisticRegression\\HorseColicTraining.txt')  # Open the training data file
    frTest = open('D:\\ML Internship\\logisticRegression\\HorseColicTest.txt')  # Open the test data file

    trainingSet = []  # Initialize a list to store the training set
    trainingLabels = []  # Initialize a list to store the training labels

    for line in frTrain.readlines():  # Read each line from the training file
        currLine = line.strip().split('\t')  # Split the line by tabs
        lineArr = []  # Initialize a list to store the features of the current sample
        for i in range(21):  # Iterate over the first 21 elements (features)
            lineArr.append(float(currLine[i]))  # Convert to float and append to lineArr
        trainingSet.append(lineArr)  # Append the feature vector to the training set
        trainingLabels.append(float(currLine[21]))  # Append the label to the training labels

    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)  # Train the model using stochastic gradient ascent with 500 iterations

    errorCount = 0  # Initialize the error count
    numTestVec = 0.0  # Initialize the number of test samples

    for line in frTest.readlines():  # Read each line from the test file
        numTestVec += 1.0  # Increment the number of test samples
        currLine = line.strip().split('\t')  # Split the line by tabs
        lineArr = []  # Initialize a list to store the features of the current sample
        for i in range(21):  # Iterate over the first 21 elements (features)
            lineArr.append(float(currLine[i]))  # Convert to float and append to lineArr

        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):  # If the predicted label is not equal to the actual label
            errorCount += 1  # Increment the error count

    errorRate = (float(errorCount) / numTestVec)  # Calculate the error rate
    print("the error rate of this test is: %f" % errorRate)  # Print the error rate
    return errorRate  # Return the error rate

def multiTest():
    numTests = 10  # Set the number of tests
    errorSum = 0.0  # Initialize the sum of errors

    for k in range(numTests):  # Run the test `numTests` times
        errorSum += colicTest()  # Accumulate the error rates from each test

    print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))  # Print the average error rate

multiTest()