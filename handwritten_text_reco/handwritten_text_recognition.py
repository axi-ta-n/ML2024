import os
from numpy import *
import operator

def classify0(inX, dataset, labels, k):
    datasetsize = dataset.shape[0]
    diffmat = tile(inX, (datasetsize, 1)) - dataset
    sqdiffmat = diffmat ** 2
    sqdist = sqdiffmat.sum(axis=1)
    dist = sqdist ** 0.5
    
    print(f"distances: {dist}")
    
    sortedDistIndices = dist.argsort()
    
    print(f"sortedDistIndices: {sortedDistIndices}")
    
    classcount = {}
    for i in range(k):
        print(f"sortedDistIndices[{i}]: {sortedDistIndices[i]}")
        
        votelabel = labels[sortedDistIndices[i]]
        print(f"votelabel: {votelabel}")
        
        classcount[votelabel] = classcount.get(votelabel, 0) + 1
        
    sortedclasscount = sorted(classcount.items(), key=operator.itemgetter(1), reverse=True)
    
    return sortedclasscount[0][0]

def img2vector(filename):
    returnvect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0,32*i+j]=int(linestr[j])
    return returnvect

testvector = img2vector('E:\\KNN data\\testDigits\\0_13.txt')
print(testvector[0,0:31])
print("\n")
print(testvector[0,32:63])

def handwritingct():
    hwlabels = []
    trainingfilelist = os.listdir('E:/KNN data/trainingDigits')  # List files in trainingDigits directory
    m = len(trainingfilelist)
    trainingmat = zeros((m, 1024))
    
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainingmat[i, :] = img2vector(f'E:/KNN data/trainingDigits/{filenamestr}')
    
    testfilelist = os.listdir('E:/KNN data/testDigits')
    errorcount = 0.0
    mtest = len(testfilelist)
    
    for i in range(mtest):
        filenamestr = testfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        vectorundertest = img2vector(f'E:/KNN data/testDigits/{filenamestr}')
        classifierResult = classify0(vectorundertest, trainingmat, hwlabels, 3)
        
        print(f"The classifier came back with: {classifierResult}, the real answer is: {classnumstr}")
        
        if classifierResult != classnumstr:
            errorcount += 1.0
    
    print(f"\nThe total number of errors is: {int(errorcount)}")
    print(f"\nThe total error rate is: {errorcount / float(mtest)}")

# Call the function to test
handwritingct()