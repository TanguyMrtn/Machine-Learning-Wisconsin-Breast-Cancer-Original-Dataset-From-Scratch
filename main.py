import csv
from random import shuffle
from math import sqrt
import matplotlib.pyplot as plt
from evaluationMeasures import *
from knn import *

def kFold(data, foldNum):
    '''
    This function split the input data in foldNum folds, with a random sampling (shuffle)
    '''
    kFolds = list()
    localData = list(data)
    shuffle(localData) # Put the data in a random order
    foldSize = int(len(data)/foldNum)
    for _ in range(foldNum):
        fold = list()
        while len(fold) < foldSize:
            fold.append(localData.pop(0))
        kFolds.append(fold)
    i = 0
    while len(localData) != 0 : # If the size of data is not a multiple of foldNum
        kFolds[i].append(localData.pop(0))
        i += 1
        if i == foldNum - 1 :
            i=0
    return kFolds

# Load the data
with open('cleaned_data.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
    data.remove(data[0])

# Number of fold (only parameter in this code)
numberOfFold = 10

# Build k-folds 
kFolds = kFold(data,numberOfFold)

# List of each fold confusion matrix (k-fold cross validation)
confusionMatrices = []

# List of each fold evaluation measures 
evaluationMeasuresList = []

currentFold = 1

# Run the k-fold cross validation
for testFold in kFolds:
    print("Fold {} out of {}".format(currentFold,numberOfFold))
    trainSet = list(kFolds)
    trainSet.remove(testFold)
    trainSet = sum(trainSet, []) # Build a large dataset composed of every fold except the current one

    expectedClasses = [sample[-1] for sample in testFold] # Get the expected class (we know it as it is in the data)
    predictedClasses, predictedProbabilityClasses = predictClasses(trainSet,testFold,3) # Do the prediction

    confusionMatrix= buildConfusionMatrix(expectedClasses,predictedClasses) # Build the confusion matrix
    confusionMatrices.append(confusionMatrix) 

    evaluationMeasures = getEvaluationMeasures(confusionMatrix,len(testFold)) # Compute the evaluation measures
    evaluationMeasuresList.append(evaluationMeasures)
        
    currentFold += 1
    print("===============================")

processAllEvaluationMeasures(evaluationMeasuresList,numberOfFold,len(data)) # Compute the overall mean of each evaluation measure

TPF,FPF = rocConfusionMetrics(expectedClasses, predictedProbabilityClasses) # Compute and plot the ROC for the last fold 
plt.plot(FPF,TPF)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()


