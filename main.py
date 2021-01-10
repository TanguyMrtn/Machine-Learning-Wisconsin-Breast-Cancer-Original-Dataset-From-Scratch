import csv
import matplotlib.pyplot as plt
from evaluationMeasures import *
from knn import *
from kfold import *
import sys

if __name__ == '__main__':

    # Number of fold and number of neighbor (parameters)
    numberOfFold = 10
    numberOfNeighbor = 3
    if len(sys.argv) == 3:
        numberOfFold=int(sys.argv[1])
        numberOfNeighbor=int(sys.argv[2])

    # Load the data
    with open('datasets/cleaned_data.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile))
        data.remove(data[0])

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
        predictedClasses, predictedProbabilityClasses = predictClasses(trainSet,testFold,numberOfNeighbor) # Do the prediction

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


