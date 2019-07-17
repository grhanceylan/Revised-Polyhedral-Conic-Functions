import os.path
import numpy as np
import PCF as pcf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

"""
This file contain experiment setup for binary classification by using PCF Algorithm
"""

#this function separates given data sets to into two subsets where, A belongs to -1 and B belongs to +1
#data:  data set, labels:  labels, indexes: indexes of the points whose label will be changed,
#l_a: the original label of the set  which will be A
def seperatetoAB(data, labels, indexes, l_a):
    A = []
    B = []
    for i in indexes:
        if labels[i] == l_a:
            A.append(data[i])
        else:
            B.append(data[i])
    A = np.array(A)
    B = np.array(B)
    return A, B



#provide data and labels 
X,lables = [],[]
labels = np.array(labels)
X = np.array(X)

#get unique labels
unique_labels = np.unique(labels)


#contains test accuracy scores
acc = []
#contains training accuracy scores
accTrain = []
#contains start and finish time
timeS = []
timeF = []
#contains number of PCFs
numPcfs=[]
skf = StratifiedKFold(n_splits=10)
for train, test in skf.split(X,labels):
      #get separated data
      sepData = seperatetoAB(X, labels, train, unique_labels[0])
      #store started time
      timeS.append(time.time())
      #create an PCF Algorithm
      pModel = pcf.PCF_iterative()
      #fit model for separated data
      pModel.fit_iter(sepData[0], sepData[1])
      #store finishing time
      timeF.append(time.time())
      #store performance metrics
      acc.append(accuracy_score(labels[test], pModel.predict_binary(X[test],unique_labels[0], unique_labels[1])))
      accTrain.append(accuracy_score(labels[train],pModel.predict_binary(X[train],unique_labels[0], unique_labels[1])))
      numPcfs.append(len(pModel.pcfs))
#Print average performance metrics
print "Time Mean",np.mean(np.subtract(timeF,timeS))
print "Av. Pcfs Count", np.mean(numPcfs), "STD:", np.std(numPcfs)
print "Train Acc.",np.mean(accTrain)
print "Test Acc.",np.mean(acc)
print "Test STD",np.std(acc)


