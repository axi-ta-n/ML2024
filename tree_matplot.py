import operator
from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:                        
        currentLabel = featVec[-1]                      
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0              
            labelCounts[currentLabel] += 1             
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)             
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels
'''


mydat[0][-1]='maybe'
print(mydat)
b = calcShannonEnt(mydat)
print(b)'''

def splitDataSet(dataset,axis,value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
'''
mydat,labels = createDataSet()
print(mydat)
a = splitDataSet(mydat,0,1)
print(a)
b = splitDataSet(mydat,0,0)
print(b)
c = splitDataSet(mydat,0,2)
print(c)
'''

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                        
        newEntropy = 0.0
        for value in uniqueVals:                            
            subDataSet = splitDataSet(dataSet, i, value)      
            prob = len(subDataSet)/float(len(dataSet))      
            newEntropy += prob * calcShannonEnt(subDataSet) 
        infoGain = baseEntropy - newEntropy                 
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain          
            bestFeature = i                   
    return bestFeature

'''
m = chooseBestFeatureToSplit(myDat)
print(m)
'''
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), 
    key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] 

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]                               
    if len(dataSet[0]) == 1:                       
        return majorityCnt(classList)                

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])                                  
    featValues = [example[bestFeat] for example in dataSet]   
    uniqueVals = set(featValues)                           
    for value in uniqueVals:
        subLabels = labels[:] 

        myTree[bestFeatLabel][value] = createTree(splitDataSet\
                          (dataSet, bestFeat, value),subLabels)
    return myTree


myDat,labels=createDataSet()
myTree = createTree(myDat,labels)
print(myTree)

# Create dataset and labels
myData, labels = createDataSet()

# Create the decision tree
tree = createTree(myData, labels[:])  # Use a copy of labels to avoid modifying the original

# Print the decision tree
print(tree)

import matplotlib.pyplot as plt

# Define the styles for decision nodes and leaf nodes
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# Function to plot a node with an arrow pointing to it
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', 
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, 
                            arrowprops=arrow_args)

# Function to calculate the number of leaf nodes in a tree
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# Function to calculate the depth of a tree
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

# Function to plot the mid-text between parent and child nodes
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

# Function to plot the tree structure
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

# Function to create the plot for a given tree
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

# Function to retrieve an example tree
def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]

# example trees
myTree = retrieveTree(0)
# createPlot(myTree)
Leafs=getNumLeafs(myTree)
Depth=getTreeDepth(myTree)
# print("Leafs: ",Leafs)
# print("Depth: ",Depth)
# myTree['no surfacing'][3]='maybe'
# createPlot(myTree)

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else: classLabel = secondDict[key]
    return classLabel

#example testing
# label=classify(myTree,labels,[1,1])
# print("Is it fish? ",label)

fr=open('D:\\ML Internship\\decision_trees\lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses,lensesLabels)
createPlot(lensesTree)
