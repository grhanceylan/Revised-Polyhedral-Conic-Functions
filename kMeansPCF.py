import numpy as np
import random
from sklearn.cluster import KMeans
from gurobipy import *

"""
   Implementation of Clustering based polyhedral conic functions algorithm in classification." Journal of Industrial & Management Optimization 11.3 (2015): 921-932.
"""



class PCF:
    def __init__(self, center):
        self.center = center

    def fit(self, c_A, c_B):
        #get the number of features
        dimension = len(c_A[0])
        # m =  s(A), p = s(B)
        m = len(c_A)
        p = len(c_B)
        #create gurobi model
        model = Model()
        model.setParam('OutputFlag', 0)
  
        #add gamma and ksi variables
        gamma = model.addVar(vtype=GRB.CONTINUOUS, lb=1)
        ksi = model.addVar(vtype=GRB.CONTINUOUS,lb=0)
        #add w variables
        w = list()
        for i in range(dimension):
            w.append( model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY))
        #add error variables
        errA = []
        for i in range(m):
            errA.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errA[%s]' % i))
        errB = []
        for i in range(p):
            errB.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='errB[%s]' % i))
        model.update()
        # add constraints
        for i in range(m):
            model.addConstr(  quicksum(w[j]*c_A[i][j] for j in range(dimension)) + ksi * np.linalg.norm( c_A[i], ord=1) - gamma  <= errA[i])
        for i in range(p):
            model.addConstr(-quicksum(w[j] * c_B[i][j] for j in range(dimension)) - ksi * np.linalg.norm(c_B[i],ord=1) + gamma  <=errB[i])

        #set objective function
        model.setObjective( (1.0/m)* quicksum(i for i in errA)+ (1.0/p)*(quicksum(i for i in errB)), GRB.MINIMIZE)
        #solve problem
        model.optimize()
        #get optimized parameters
        self.gamma = gamma.X
        self.ksi = ksi.X
        self.w = np.zeros(dimension)
        for i in range(dimension):
            self.w[i]= w[i].X

class PCF_iterative:
    def __init__(self):
        #generated PCFs
        self.pcfs = list()
        #parameters

    def kmeans(self, dt,k):
        result = KMeans(n_clusters=k,n_init=1).fit(dt)
        return result.cluster_centers_,result.labels_



    def fit_iter(self,A,B,k=2):
        # do until A or B become an empty set
            centers,clusters= self.kmeans(A,k)
            for ki in range(k):
                c_A=np.subtract(A,centers[ki])
                c_B=np.subtract(B, centers[ki])
                # create a model
                temp = PCF(centers[ki])
                # solve the sub-problem
                temp.fit( c_A[clusters==ki], c_B)
                # store obtained pcf
                self.pcfs.append(temp)
        

            # return generated PCFs
            return self.pcfs

    #binary prediction
    def predict_binary(self, X, labelA, labelB):
        predictions = list()
        for i in range(len(X)):
            f = 0
            for p in self.pcfs:
                f = np.dot(np.subtract(X[i], p.center), p.w) + p.ksi * np.linalg.norm(np.subtract(X[i], p.center),ord=1) - p.gamma
                if f <= 0.0:
                    f = labelA
                    break
                else:
                    f = labelB
            predictions.append(f)
        return predictions

    # multi label prediction
    def predict_multi(self, X, pcflist, labels):
        predictions = []
        for i in range(len(X)):
            f = 0
            minValues=[]
            for plist in pcflist:
                values=[]
                for p in plist:
                    f = np.dot(np.subtract(X[i], p.center), p.w) + p.ksi* np.linalg.norm( np.subtract(X[i], p.center), ord=1) - p.gamma
                    values.append(f)
                minValues.append(np.amin(values))
            predictions.append(labels[ np.argmin(minValues)])
        return predictions






