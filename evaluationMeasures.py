from math import sqrt

def buildConfusionMatrix(expected, predicted):
    '''
    This function build the confusion matrix, based on the expected and predicted classes
    '''
    matrix = [[0,0],[0,0]] # Binary classification problem
    for currentPrediction, expectedPrediction in zip(predicted, expected):
        matrix[int(currentPrediction)][int(expectedPrediction)] += 1 
    matrix.reverse()
    matrix[0].reverse()
    matrix[1].reverse()
    return matrix

def rocConfusionMetrics(expected, predictions):
    '''
    This function computes the True and False Positive Rates with a threshold for ploting the ROC curve
    '''
    TPR = []
    FPR = []
    i = 0
    threshold_step = 0.0002 # Threshold of the ROC computation
    while (i <= 1): 
        threshold = i # For each threshold we compute the True and False Positive Rates 
        TP=0
        FP=0
        TN=0
        FN=0
        for j in range(0,len(expected)): 
        
            if int(expected[j])==1: # We check the predicted probability with the threshold
                if int(predictions[j])>=threshold:
                    TP += 1
                else:
                    FN += 1
            else:
                if int(predictions[j])>=threshold:
                    FP += 1
                else:
                    TN += 1
        truePositiveRate = TP / (TP + FN)
        falsePositiveRate = FP / (FP + TN)

        TPR.append(truePositiveRate)
        FPR.append(falsePositiveRate)
        i += threshold_step
    TPR.reverse()
    FPR.reverse()
    return (TPR,FPR)

def getEvaluationMeasures(confusionMatrix,numberOfSamples):
    '''
    This function computes some evaluation measures based on the confusion matrix
    accuracy (and 95% confidence interval), precision, sensitivity, specificity, f_score
    '''
    TP = confusionMatrix[0][0]
    FP = confusionMatrix[0][1]
    FN = confusionMatrix[1][0]
    TN = confusionMatrix[1][1]
    accuracy = round((TP + TN) / (TP + TN + FP + FN),3)
    confidenceInterval = round(1.96*sqrt((accuracy*(1-accuracy))/numberOfSamples),3) # 95% confidence interval
    precisionPositive = round(TP / (TP + FP),3)
    precisionNegative = round(TN / (TN + FN),3)

    sensitivity = round(TP / (TP + FN),3)
    specificity = round(TN / (TN + FP),3)

    f_scorePositive = round(2 / ((1 / precisionPositive) + (1 / sensitivity)),3)
    f_scoreNegative = round(2 / ((1 / precisionNegative) + (1 / specificity)),3)
    f_score = round((f_scorePositive + f_scoreNegative) / 2,3)

    print("Accuracy : {accuracy} +/- {interval} (95 percent confidence interval)".format(accuracy=accuracy,interval=confidenceInterval))
    print("Positive prediction value : {}".format(precisionPositive))
    print("Negative prediction value : {}".format(precisionNegative))
    print("Sensitivity : {}".format(sensitivity))
    print("Specificity : {}".format(specificity))
    print("f-score : {}".format(f_score))

    return (accuracy,precisionPositive,precisionNegative,sensitivity,specificity,f_score)

def processAllEvaluationMeasures(evaluationMeasures,numberOfFold,dataSize):
    '''
    This function computes the mean of each evaluation measures obtained for each fold
    '''
    meanAccuracy = 0
    meanPrecisionPositive = 0
    meanPrecisionNegative = 0
    meanSensitivity = 0
    meanSpecificity = 0
    meanFscore = 0
    for evaluationMeasure in evaluationMeasures:
        meanAccuracy += evaluationMeasure[0]
        meanPrecisionPositive += evaluationMeasure[1]
        meanPrecisionNegative += evaluationMeasure[2]
        meanSensitivity += evaluationMeasure[3]
        meanSpecificity += evaluationMeasure[4]
        meanFscore += evaluationMeasure[5]
    meanAccuracy = round(meanAccuracy / numberOfFold,3)
    confidenceInterval = round(1.96*sqrt((meanAccuracy*(1-meanAccuracy))/dataSize),3)
    meanPrecisionPositive = round(meanPrecisionPositive / numberOfFold,3)
    meanPrecisionNegative = round(meanPrecisionNegative / numberOfFold,3)
    meanSensitivity = round(meanSensitivity / numberOfFold,3)
    meanSpecificity = round(meanSpecificity / numberOfFold,3)
    meanFscore = round(meanFscore / numberOfFold,3)

    print("Mean Accuracy : {accuracy} +/- {interval} (95 percent confidence interval)".format(accuracy=meanAccuracy,interval=confidenceInterval))
    print("Mean Positive prediction value : {}".format(meanPrecisionPositive))
    print("Mean Negative prediction value : {}".format(meanPrecisionNegative))
    print("Mean Sensitivity : {}".format(meanSensitivity))
    print("Mean Specificity : {}".format(meanSpecificity))
    print("Mean f-score : {}".format(meanFscore))