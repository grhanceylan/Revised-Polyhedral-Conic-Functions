import os.path
import numpy as np
import PCF as pc
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


#contains generated PCFs
pcfs=[]
#contains start and finish  time
timeS =[]
timeF =[]
#for each unique label apply PCF algorithm
for lbl in uniqueLabels:
    sepData = seperatetoAB(X, labels, train, lbl)
    timeS.append(time.time())
    pModel = pc.PCF_iterative()
    pModel.fit_iter(sepData[0],sepData[1])
    timeF.append(time.time())
    pcfs.append(pModel.pcfs)

#create an empty model for prediction
accModel=pc.PCF_iterative()
#print performance metrics
numbPcfs = sum([len( pcfs[i]) for i in range(len(uniqueLabels))])
print "\tAvg. Numb. of PCFs", numbPcfs, "\tSTD:", np.std(numbPcfs)
print "\tTraining Time:", round(sum(timeF)-sum(timeS),2)
predictions = accModel.predict_multi(X[train], pcfs, uniqueLabels)
trainingAcc = accuracy_score(labels[train], predictions)
print "\tTraining Acc:", trainingAcc
predictions = accModel.predict_multi(X[test], pcfs,uniqueLabels)
acc = accuracy_score(labels[test], predictions)
print "\tTest Acc:", acc, "\tSTD:", np.std(acc)


