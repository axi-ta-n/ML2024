import numpy as np
import matplotlib.pyplot as plt
import random

# Function to load the dataset
def loadDataSet():
    dataMat = []  # Initialize an empty list to store feature vectors
    labelMat = []  # Initialize an empty list to store labels
    with open('D:\\ML Internship\\logisticRegression\\testSet.txt') as fr:  # Open the dataset file
        for line in fr.readlines():  # Iterate over each line in the file
            lineArr = line.strip().split()  # Strip whitespace and split the line by spaces
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # Append a feature vector (with bias term 1.0) to dataMat
            labelMat.append(int(lineArr[2]))  # Append the label to labelMat
    return dataMat, labelMat  # Return the feature vectors and labels

# Sigmoid function to map any real value between 0 and 1
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))  # Return the sigmoid of inX

# Gradient ascent function to optimize weights
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # Convert the input list to a NumPy matrix
    labelMat = np.mat(classLabels).transpose()  # Convert the class labels to a column vector
    m, n = np.shape(dataMatrix)  # Get the dimensions of the data matrix (m: number of samples, n: number of features)
    alpha = 0.001  # Set the learning rate
    maxCycles = 500  # Set the maximum number of iterations
    weights = np.ones((n, 1))  # Initialize the weight vector with ones

    for k in range(maxCycles):  # Iterate maxCycles times
        h = sigmoid(dataMatrix * weights)  # Compute the hypothesis (predicted values)
        error = labelMat - h  # Compute the error (difference between actual and predicted values)
        weights = weights + alpha * dataMatrix.transpose() * error  # Update the weights using gradient ascent

    return np.array(weights)  # Return the optimized weights as a NumPy array

# Function to plot the dataset and the decision boundary
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()  # Load the dataset
    dataArr = np.array(dataMat)  # Convert the list of lists to a NumPy array
    n = np.shape(dataArr)[0]  # Get the number of data points (rows)
    xcord1 = []  # Initialize list to store x-coordinates of class 1
    ycord1 = []  # Initialize list to store y-coordinates of class 1
    xcord2 = []  # Initialize list to store x-coordinates of class 0
    ycord2 = []  # Initialize list to store y-coordinates of class 0

    for i in range(n):  # Iterate over all data points
        if int(labelMat[i]) == 1:  # If the label is 1 (class 1)
            xcord1.append(dataArr[i, 1])  # Append the x-coordinate to xcord1
            ycord1.append(dataArr[i, 2])  # Append the y-coordinate to ycord1
        else:  # If the label is 0 (class 0)
            xcord2.append(dataArr[i, 1])  # Append the x-coordinate to xcord2
            ycord2.append(dataArr[i, 2])  # Append the y-coordinate to ycord2

    fig = plt.figure()  # Create a new figure
    ax = fig.add_subplot(111)  # Add a subplot to the figure
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  # Plot class 1 points in red
    ax.scatter(xcord2, ycord2, s=30, c='green')  # Plot class 0 points in green

    x = np.arange(-3.0, 3.0, 0.1)  # Create an array of x values from -3.0 to 3.0 with a step of 0.1
    y = (-weights[0] - weights[1] * x) / weights[2]  # Calculate the corresponding y values for the decision boundary
    ax.plot(x, y.ravel())  # Plot the decision boundary

    plt.xlabel('X1')  # Set the x-axis label
    plt.ylabel('X2')  # Set the y-axis label
    plt.show()  # Display the plot

# Load dataset and calculate weights using gradient ascent
dataArr, labelMat = loadDataSet()
weights = gradAscent(dataArr, labelMat)
print(weights)  # Print the optimized weights
plotBestFit(weights)  # Plot the dataset and decision boundary


# Stochastic Gradient Ascent (single iteration over data points)
def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)  # Get the number of samples (m) and number of features (n)
    alpha = 0.01  # Set the learning rate
    weights = np.ones(n)  # Initialize the weights vector with ones

    for i in range(m):  # Iterate over each sample in the dataset
        h = sigmoid(np.dot(dataMatrix[i], weights))  # Compute the hypothesis (predicted value) for sample i
        error = classLabels[i] - h  # Calculate the error (difference between actual and predicted value)
        weights = weights + alpha * error * dataMatrix[i]  # Update the weights using the stochastic gradient ascent rule

    return weights  # Return the optimized weights

# Stochastic Gradient Ascent (multiple iterations over data points)
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)  # Get the number of samples (m) and number of features (n)
    weights = np.ones(n)  # Initialize the weights vector with ones

    for j in range(numIter):  # Iterate over the number of iterations
        dataIndex = list(range(m))  # Create a list of indices for the data points
        for i in range(m):  # Iterate over each sample in the dataset
            alpha = 4 / (1.0 + j + i) + 0.01  # Set the learning rate, which decreases with each iteration
            randIndex = int(random.uniform(0, len(dataIndex)))  # Select a random index from the remaining data points
            h = sigmoid(np.dot(dataMatrix[randIndex], weights))  # Compute the hypothesis (predicted value) for the randomly selected sample
            error = classLabels[randIndex] - h  # Calculate the error (difference between actual and predicted value)
            weights = weights + alpha * error * dataMatrix[randIndex]  # Update the weights using the stochastic gradient ascent rule
            del(dataIndex[randIndex])  # Remove the selected index from the list of indices to avoid re-selection in the same iteration

    return weights  # Return the optimized weights

'''
dataArr, labelMat = loadDataSet()
weights = stocGradAscent0(np.array(dataArr), labelMat)
print(weights)
plotBestFit(weights)
'''