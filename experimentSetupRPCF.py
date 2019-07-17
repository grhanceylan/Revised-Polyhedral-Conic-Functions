import os.path
import numpy as np
import PCF as pcf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time



"""
This file contain experiment setup for binary classification by using r-PCF Algorithm
"""
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




#penalty parameter set
c=[-4,-3,-2,-1,0,1,2,3,4]
#regularization parameter set
l=[-4,-3,-2,-1,0,1,2,3,4]

skf = StratifiedKFold(n_splits=10)
for ci in c:
     print "********** Penalty: ",ci
     for li in l:
        print "*** Regularization: ", li
        acc = []
        timeS = []
        timeF = []
        accTrain = []
        numPcfs=[]
        for train, test in skf.split(X,labels):
                    sepData = seperatetoAB(X, labels, train, unique_labels[0])
                    timeS.append(time.time())
                    pModel = pcf.PCF_iterative()
                    pModel.fit_iter(sepData[0], sepData[1])
                    timeF.append(time.time())
                    acc.append(accuracy_score(labels[test], pModel.predict_binary(X[test],unique_labels[0], unique_labels[1])))
                    accTrain.append(accuracy_score(labels[train],pModel.predict_binary(X[train],unique_labels[0], unique_labels[1])))
                    numPcfs.append(len(pModel.pcfs))
        print "Time Mean",np.mean(np.subtract(timeF,timeS))
        print "Av. Pcfs Count", np.mean(numPcfs), "STD:", np.std(numPcfs)
        print "Train Acc.",np.mean(accTrain)
        print "Test Acc.",np.mean(acc)
        print "Test STD",np.std(acc)


