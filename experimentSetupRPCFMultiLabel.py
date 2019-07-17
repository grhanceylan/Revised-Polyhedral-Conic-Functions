import os.path
import numpy as np
import rPCF as pc
from sklearn.metrics import accuracy_score
import time
import random

#this function separates given data sets to into two subset where, A belongs to -1 and B belongs to +1
#data: whole data set, labels: whole labels, indexes: indexes of the points whose label will be changed,
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
#provide number of training samples
numb_train=?
train = random.sample(range(len(X)), numb_train)
test = [ i for i in range(len(X)) if i not in train]
#get unique labels
uniqueLabels = np.unique(labels)



#penalty parameter set
c=[-4,-3,-2,-1,0,1,2,3,4]
#regularization parameter set
l=[-4,-3,-2,-1,0,1,2,3,4]

for ci in c:
     print "********** Penalty: ",ci
     for li in l:
        print "*** Regularization: ", li
        pcfList=[]
        timeS =[]
        timeF =[]
        #apply r-PCF Algorithm for each unique labels
        for lbl in uniqueLabels:
             print "* Label: ", lbl
             sepData = seperatetoAB(X, labels, train, lbl)
             timeS.append(time.time())
             pModel = pc.PCF_iterative(10**ci,10**li)
             pModel.fit_iter(sepData[0],sepData[1])
             timeF.append(time.time())
             pcfList.append(pModel.pcfs)


        accModel=pc.PCF_iterative(0,0)
        numbPcfs = sum([len( pcfList[i]) for i in range(len(uniqueLabels))])


        print "\tAvg. Numb. of PCFs", numbPcfs, "\tSTD:", np.std(numbPcfs)
        print "\tTraining Time:", round(sum(timeF)-sum(timeS),2)
        predictions = accModel.predict_multi(X[train], pcfList, uniqueLabels)
        trainingAcc = accuracy_score(labels[train], predictions)
        print "\tTraining Acc:", trainingAcc
        predictions = accModel.predict_multi(X[test], pcfList,uniqueLabels)
        acc = accuracy_score(labels[test], predictions)
        print "\tTest Acc:", acc, "\tSTD:", np.std(acc)


