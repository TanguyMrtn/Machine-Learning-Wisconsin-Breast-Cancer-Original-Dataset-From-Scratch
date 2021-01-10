from random import shuffle

def kFold(data, foldNum):
    '''
    This function split the input data in foldNum folds, with a random sampling (shuffle)
    '''
    kFolds = list()
    localData = list(data)
    shuffle(localData) # Put the data in a random order
    foldSize = int(len(data)/foldNum)
    for __temp in range(foldNum):
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