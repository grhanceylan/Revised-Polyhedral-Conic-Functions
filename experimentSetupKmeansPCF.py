import os.path
import numpy as np
import kMeansPCF as pc
from sklearn.metrics import accuracy_score
import time
import random


"""
   Experiment setup  of k-means PCF Algorithm
"""
#this function separates given data sets to into two subsets where, A belongs to -1 and B belongs to +1
#data:  data set, labels:  labels, indexes: indexes of the points whose label will be changed,
#l_a: original label of the set  which will be A
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
#provide number of training samples
numb_train=?
train = random.sample(range(len(X)), numb_train)
test = [ i for i in range(len(X)) if i not in train]
#get unique labels
uniqueLabels = np.unique(labels)


#k parameter set
k=[2,3,4,5,6,7,8,9,10,11,12]

for ki in k:
            print "********** Cluster: ",ki
            #contains start and finish  time
            timeS =[]
            timeF =[]
            #contains generated PCFs
            pcfs=[]
            #for each unique label apply k-means PCF algorithm
            for lbl in uniqueLabels:
                sepData = seperatetoAB(X, labels, train, lbl)
                timeS.append(time.time())
                pModel = pc.PCF_iterative()
                pModel.fit_iter(sepData[0],sepData[1],k=ki)
                timeF.append(time.time())
                pcfs.append(pModel.pcfs)
             #create an empty model for prediction
            accmodel= pc.PCF_iterative()
            print "\tAvg. Numb. of PCFs", sum([len(pcfs[i]) for i in range(len(uniqueLabels))])
            print "\tTraining Time:", round(sum(timeF)-sum(timeS),2)
            predictions = accmodel.predict_multi(X[train], pcfs, uniqueLabels)
            trainingAcc = accuracy_score(labels[train], predictions)
            print "\tTraining Acc:", trainingAcc
            predictions = accmodel.predict_multi(X[test], pcfs,uniqueLabels)
            acc = accuracy_score(labels[test], predictions)
            print "\tTest Acc:", acc, "\tSTD:", np.std(acc)


