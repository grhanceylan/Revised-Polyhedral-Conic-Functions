import numpy as np
import random
from gurobipy import *

"""
   
"""


class PCF:
    def __init__(self, center):
        self.center = center

    def fit(self, c_A, c_B, C, L):
        #get the number of features
        dimension = len(c_A[0])
        # m =  s(A), p = s(B)
        m = len(c_A)
        p = len(c_B)
        #create gurobi model
        model = Model()
        model.setParam('OutputFlag', 0)
       # model.setParam('Method',2)
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
            model.addConstr(  quicksum(w[j]*c_A[i][j] for j in range(dimension)) + ksi * np.linalg.norm( c_A[i], ord=1) - gamma + 1 <= errA[i])
        for i in range(p):
            model.addConstr(-quicksum(w[j] * c_B[i][j] for j in range(dimension)) - ksi * np.linalg.norm(c_B[i],ord=1) + gamma + 1 <=errB[i])

        #set objective function
        model.setObjective( quicksum(i for i in errA)+ C* (quicksum(i for i in errB))+ L*( quicksum(w[j]*w[j] for j in range(dimension))+ksi*ksi+gamma*gamma), GRB.MINIMIZE)
        #solve problem
        model.optimize()
        #get optimized parameters
        self.gamma = gamma.X
        self.ksi = ksi.X
        self.w = np.zeros(dimension)
        for i in range(dimension):
            self.w[i]= w[i].X

class PCF_iterative:
    def __init__(self,C,L):
        #generated PCFs
        self.pcfs = list()
        #parameters
        self.C=C
        self.L=L

    def fit_iter(self, A, B):
        # do until A or B become an empty set
        while len(A)*len(B) !=0:
            # select a random center from set A
            r = random.randint(0, len(A) - 1)
            # set a_k=a_r
            center=A[r]
            # ceterilize given sets with a_k
            c_A, c_B = np.subtract(A, center), np.subtract(B, center)

            # create a model
            temp = PCF(center)
            # solve the sub-problem
            temp.fit(c_A, c_B, self.C, self.L)
            # store obtained pcf
            self.pcfs.append(temp)
            # update set A
            cnt1 = []
            for i in range(len(A)):

                if  np.dot( c_A[i], temp.w) + temp.ksi * np.linalg.norm(c_A[i],ord=1) - temp.gamma > 0.0:
                    cnt1.append(i)
            A = A[cnt1]
            # update set B
            cnt2 = []
            for i in range(len(B)):
                if np.dot( c_B[i], temp.w) + temp.ksi * np.linalg.norm(c_B[i],ord=1) - temp.gamma > 0.0:
                    cnt2.append(i)

            B = B[cnt2]

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






