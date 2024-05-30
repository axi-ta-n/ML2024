from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

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

# Testing the function
group, labels = createDataSet()
result = classify0([3, 1], group, labels, 3)
print(result)  
