from math import sqrt

def euclideanDistance(firstSample, secondSample):
    '''
    This function computes the euclidean distance between two samples
    '''
    distance = 0
    for i in range(0,len(firstSample)-1): # -1 because the last value of the list is the output class
    	distance += (int(firstSample[i]) - int(secondSample[i]))**2
    return sqrt(distance)

def getNeighbors(trainData, testData, k):
    '''
    This function gets the k nearest neighbors of the 'testData' sample
    '''
    distances = list()
    for trainSample in trainData: # We compute each euclidian distance between the train samples (the ones in all the other folds) and the current test sample
        distance = euclideanDistance(testData, trainSample)
        distances.append((trainSample, distance))
    distances.sort(key=lambda x: x[1]) # We sort the list
    neighbors = list()
    for i in range(k): # We take the k nearest one
    	neighbors.append(distances[i][0])
    return neighbors

def predictClasses(trainData, testData, k):
    '''
    This function predicts the class of the samples in 'testData', the neighbors are the samples in trainData
    '''
    predictions = list()
    probabilityPredictions = list()
    for i in range(len(testData)):
        neighbors = getNeighbors(trainData, testData[i], k) # We get the k-nearest neighbors of the current test sample
        neighbors_classes = [neighbor[-1] for neighbor in neighbors] # We get the class of each neighbor
        predictions.append(max(set(neighbors_classes), key=neighbors_classes.count)) # The predicted class is the one that is taken by most of the neighbors
        probabilityPrediction = 0 # We also compute the probability for the ROC curve
        for neighborClass in neighbors_classes :
            probabilityPrediction += int(neighborClass)
        probabilityPredictions.append(probabilityPrediction/len(neighbors_classes))
    return (predictions,probabilityPredictions)