
'''
This file uses a perceptron algorithm and works with binary classification to test four different datasets

By looking at the results, it gives a idea that 822179_4 contains much more data that is linearly separable. Other three files
contain less linear data.

'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

def findResults(features,label,nCol,nRow,results):
    '''
    This function finds out activation, compares it to threshold and finds out if the weights or threshold should be adjusted
    as per the output received. It then returns an array that contains adjusted weights, adjusted threshold, the result of 
    prediction as if the algorithm guessed the label right or wrong
    '''
    outputArray = np.zeros((nRow,int(1)),float)
    iRow = 0
    iCol = 0
    output = 0
    while iRow < nRow:
        a = 0
        t = results[iRow,nCol-2]
        while iCol < nCol-2:
            w = results[iRow,iCol]
            a += features.iloc[iRow,iCol]*w
            iCol += 1
        if(a < t or a==t):
            output = 0.0
        else:
            output = 1.0
        if(output != label.iloc[iRow]):
            results[iRow,nCol-1] = 0.0
        else:
            results[iRow,nCol-1] = 1.0
            
        #If the algorithm made the wrong label guess then the weights and threshold of that record should be adjusted
        if(results[iRow,nCol-1] == 0.0):
            col = 0
            if(output == 0.0):
                while col < nCol-2:
                    if(results[iRow,col] < 0.1):
                        results[iRow,col] += 0.05
                    col += 1
                results[iRow,nCol-2] -= 0.05
            elif(output == 1.0):
                while col < nCol-2:
                    if(results[iRow,col] >= 0.1):
                        results[iRow,col] -= 0.05
                    col += 1
                results[iRow,nCol-2] += 0.05
        iRow += 1
    return results
n = 1
#Perform four iterations to go over four datasets
while n < 5:
    print()
    print("************** Dataset ",n," ****************")
    nStr = str(n)
    print("File: 822179_"+nStr+".csv")
    data = pd.read_csv('822179_'+nStr+'.csv',header=None)
    n += 1
    nCol = len(data.columns)-1
    features = data.iloc[:,0:nCol]
    label = data.iloc[:,nCol]
    X_train, X_test, y_train, y_test = train_test_split(features, label)
    percep = Perceptron()
    percep.fit(X_train,y_train)
    y_pred = percep.predict(X_test)
    accuracy = round((accuracy_score(y_pred,y_test))*100,2)
    nRow = len(X_train)
    results = np.zeros((nRow,int(nCol+2)),dtype=float)
    epoch = 0
    
    # Do not stop iterating until all the predicted labels match original ones
    while len(X_train) > 0:
        results = findResults(X_train,y_train,nCol+2,nRow,results)
        X_train = X_train[results[:,nCol+1]==0.0]
        y_train = y_train[results[:,nCol+1]==0.0]
        nRow = len(X_train)
        results = results[results[:,nCol+1]==0.0]
        epoch += 1
    nRow = len(X_test)
    results = np.zeros((nRow,int(nCol+2)),dtype=float)
    epoch = 0
    finalW = results
    while len(X_test) > 0:
        results = findResults(X_test,y_test,nCol+2,nRow,results)
        X_test = X_test[results[:,nCol+1]==0.0]
        y_test = y_test[results[:,nCol+1]==0.0]
        nRow = len(X_test)
        finalW = results[0,0:7]
        finalT = results[0,7]
        results = results[results[:,nCol+1]==0.0]
        epoch += 1
    print("Accuracy: ",accuracy)
    print("W: ",finalW)
    print("T: ",finalT)
    print("Epoch: ",epoch)

