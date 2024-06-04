import numpy
import math
'''Word list to vector'''

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', \
                  'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', \
                  'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', \
                   'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
                   'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec


def createVocabList(dataset):
    vocabSet = set([])
    for documnet in dataset:
        vocabSet = vocabSet | set(documnet) #to select only unique values
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)             
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1

        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec
    '''takes the vocabulary list and a document and outputs a vec
tor of 1s and 0s to represent whether a word from our vocabulary is present or not in
 the given document'''

listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print(myVocabList)

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = numpy.ones(numWords)  # Initialize counts to 1 for Laplace smoothing
    p1Num = numpy.ones(numWords)  # Initialize counts to 1 for Laplace smoothing
    p0Denom = 2.0  # Initialize denominator to 2 for Laplace smoothing
    p1Denom = 2.0  # Initialize denominator to 2 for Laplace smoothing
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1vect = numpy.log(p1Num / p1Denom)
    p0vect = numpy.log(p0Num / p0Denom)
    return p0vect, p1vect, pAbusive


trainMat=[]
for PostinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,PostinDoc))

p0V,p1V,pAbuse=trainNB0(trainMat,listClasses)
print("Probability of abusive sentence:",pAbuse)
print("P0 vector:\n",p0V)
print("P1 vector:\n",p1V)
'''takes a matrix of documents, trainMatrix, and a vector
 with the class labels for each of the documents, trainCategory. 
 Every time a word appears in a document, the count for that word 
 (p1Num or p0Num) gets incremented, and the total number 
 of words for a document gets summed up over all the documents. C You do this
 for both classes. 
 Finally, you divide every element by the total number of words for that class.'''

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)         
    p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    listofPosts,listClasses=loadDataSet()
    myvocablist=createVocabList(listofPosts)
    trainMat=[]
    for postinDoc in listofPosts:
        trainMat.append(setOfWords2Vec(myvocablist,postinDoc))
    p0V,p1V,pAb = trainNB0(numpy.array(trainMat),numpy.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = numpy.array(setOfWords2Vec(myvocablist, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = numpy.array(setOfWords2Vec(myvocablist, testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


testingNB()