import csv
import random
import math
import operator
from random import shuffle

# Assumption if one of the columns in header is a string then csv has header or it has data from first row.
def getColumns( filename, column_header=[] ):
    filereader = csv.reader( open( filename ) )
    row_counter = 0

    for row in filereader :
        isString = True
        if row_counter == 0:
            row_header = row
            colnum = 0
            for col in row_header:
                try :
                    float(col)
                    isString = False
                except :
                    isString = True
                if ( isString ) :
                    column_header.append( row_header[colnum] )
                    colnum += 1
        break
    # print column_header
    return column_header

def readData( filename, split, column_header, trainDataSet=[], testDataSet=[] ):
    row_count = 0
    with open( filename, 'rb' ) as inputfile:
        lines = csv.reader(inputfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            if not column_header:
                for y in range(len(column_header)-1):
                    dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainDataSet.append(dataset[x])
                else:
                    testDataSet.append(dataset[x])
            else :
                if ( row_count != 0 ) :
                    for y in range(len(column_header)-1):
                        dataset[x][y] = float(dataset[x][y])
                    if random.random() < split:
                        trainDataSet.append(dataset[x])
                    else:
                        testDataSet.append(dataset[x])
            row_count += 1

def kFoldCrossValidation( kFold, filename, column_header ) :
    rowCount = 0
    with open( filename, 'rb' ) as inputfile:
        lines = csv.reader(inputfile)
        dataset = list(lines)
        if not column_header:
            rowCount = len(dataset)
        else :
            rowCount = len(dataset) - 1

    shuffle(dataset)
    validation_dict = {}
    counter = 1

    step = rowCount // kFold
    for i in range(0, rowCount - (rowCount%kFold), step):
        validation_dict[counter] = dataset[i:i+step]
        counter += 1
        current = i
    validation_dict[counter-1].extend(dataset[current+step:])
    return validation_dict

def euclideanDistance( testdataInstanceAlpha, traindataInstanceBeta, length ) :
    distanceEuclidean = 0
    distanceHamming = 0
    for x in range( length ):
        try :
            distanceEuclidean += pow((float(testdataInstanceAlpha[x]) - float(traindataInstanceBeta[x])), 2)
        except :
            distanceHamming += hammingDistance( traindataInstanceBeta[x], testdataInstanceAlpha[x] )

    distance = math.sqrt(distanceEuclidean) + distanceHamming
    return distance

def hammingDistance( trainInstanceValue, testInstanceValue ) :
    if ( trainInstanceValue == testInstanceValue) :
        return 0
    else :
        return 1

def nearestNeighbour(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def classifyTestInstance( neighbors ) :
    neighboursVote = {}
    for x in range( len( neighbors ) ):
        classValue = neighbors[x][-1]
        if classValue in neighboursVote:
            neighboursVote[classValue] += 1
        else:
            neighboursVote[classValue] = 1

    testInstancevotes = sorted(neighboursVote.iteritems(), key=operator.itemgetter(1), reverse=True)[0][0]
    if ( isinstance( testInstancevotes, (int, long) ) ) :
        return testInstancevotes[0][0]
    else :
        return testInstancevotes


def computeAccuracy( testDataSet, prediction ) :
    accurateCounter = 0
    for x in range(len(testDataSet)):
        if testDataSet[x][-1] == prediction[x]:
            accurateCounter += 1
    return (accurateCounter/float(len(testDataSet))) * 100.0

def averageAccuracy( accuracyList ) :
    totalAccuracy = 0
    for x in range(len(accuracyList)):
        totalAccuracy += accuracyList[x]

    return totalAccuracy/len(accuracyList)

def myknnregressor(filename, kFoldValue, kvalue):
    filename = filename;
    column_header = []
    validation_dict ={}
    kFold = kFoldValue
    getColumns( filename, column_header )

    validation_dict = kFoldCrossValidation(kFold, filename, column_header )
    accuracyList = []
    rotationCount = 1
    for k in range( kFold ) :
        testDataSet = []
        trainDataSet = []
        for key in validation_dict :
            if ( rotationCount == key ) :
                testDataSet.extend( validation_dict[key] )
            else :
                trainDataSet.extend( validation_dict[key] )
        rotationCount += 1

        prediction = []
        k=kvalue
        regressionResult = []
        for val in range(len(testDataSet)):
            neighbours = nearestNeighbour( trainDataSet, testDataSet[val], k )
            sumNeighbours = 0
            for i in range(len(neighbours)):
                if filename == 'data/Iris.csv':
                    if neighbours[i][-1] == 'Iris-versicolor':
                        sumNeighbours += 1
                    if neighbours[i][-1] == 'Iris-virginica':
                        sumNeighbours += 2
                    if neighbours[i][-1] == 'Iris-setosa':
                        sumNeighbours += 3
                else :
                    sumNeighbours += int(neighbours[i][-1])
            regressionResult.append( sumNeighbours / len(neighbours) )


        correctResult = 0
        for i in range(len(testDataSet)):
            predicted = 0
            if (filename == 'data/Iris.csv'):
                if testDataSet[i][-1] == 'Iris-versicolor':
                    predicted = 1
                if testDataSet[i][-1] == 'Iris-virginica':
                    predicted = 2
                if testDataSet[i][-1] == 'Iris-setosa':
                    predicted = 3
                print 'Predicted : ' + str(regressionResult[i]) + '     Actual : '+str(testDataSet[i][-1])
                if (str(predicted) == str(regressionResult[i])):
                    correctResult += 1
            else :
                print 'Predicted : ' + str(regressionResult[i]) + '     Actual : '+str(testDataSet[i][-1])
                if str(regressionResult[i]) == str(testDataSet[i][-1]):
                    correctResult += 1

        accuracyList = []
        accuracy = float(correctResult / len(testDataSet)) * 100.0 

        accuracyList.append( accuracy )
        print '**********************************'

    sumAccuracy = 0
    for i in range(len(accuracyList)):
        sumAccuracy += accuracyList[i]

    averageAccuracy = float(sumAccuracy) / float(len(accuracyList) )
    print 'Accuracy : ' + str(averageAccuracy)
myknnregressor( 'data/glass.data', 5, 2 )
