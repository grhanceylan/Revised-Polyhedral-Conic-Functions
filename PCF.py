import numpy as np
import random
from gurobipy import *

"""
    -Seperation via polyhedral conic functions, Gasimov and Ozturk, 2006
    -To execute this algortihm Gurobi solver and gurobi.py are required
     http://www.gurobi.com/
     https://www.gurobi.com/documentation/6.5/quickstart_mac/the_gurobi_python_interfac.html
"""


#subproblem solvers
class PCF:
    def __init__(self, center):
        #center of a PCF
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
        ksi = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
        #add w variables
        w = list()
        for i in range(dimension):
            w.append(model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY))
        #add error variables
        err_A = list()
        for i in range(m):
            err_A.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='err[%s]' % i))

        model.update()
        # add  soft constraints
        for i in range(m):
            model.addConstr(quicksum(w[j]*c_A[i][j] for j in range(dimension)) + ksi* np.linalg.norm(c_A[i],ord=1) - gamma + 1 <= err_A[i])
        # add  hard constraints
        for i in range(p):
            model.addConstr(quicksum(-w[j] * c_B[i][j] for j in range(dimension)) - ksi * np.linalg.norm(c_B[i],ord=1) + gamma + 1 <= 0)

        #set objective function
        model.setObjective( quicksum(i for i in err_A)/len(err_A), GRB.MINIMIZE)
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
        # to keep generated PCFs
        self.pcfs = list()
    def fit_iter(self, A, B):
        #do until A become an empty set
        while len(A) !=0:
            # select a random center from set A
            r = random.randint(0, len(A) - 1)
            # set a_k=a_r
            center= A[r]
            #ceterilize given sets with a_k
            c_A, c_B = np.subtract(A, center), np.subtract(B, center)
            # create a model
            temp = PCF(center)
            # solve the sub-problem
            temp.fit(c_A, c_B)
            # store obtained pcf
            self.pcfs.append(temp)
            # update set A
            cnt = []
            for i in range(len(A)):
                if np.dot(c_A[i],temp.w) + temp.ksi* np.linalg.norm(c_A[i], ord=1) - temp.gamma > 0.0:
                    cnt.append(i)
            A = A[cnt]
        #return generated PCFs
        return self.pcfs



    # binary prediction
    def predict_binary(self, X,l1,l2):
        predictions = list()
        for i in range(len(X)):
            f = 0
            for p in self.pcfs:
                f = np.dot(np.subtract(X[i], p.center), p.w) + p.ksi* np.linalg.norm(np.subtract(X[i], p.center), ord=1) - p.gamma
                if f <= 0.0:
                    f = l1
                    break
                else:
                    f = l2
            predictions.append(f)
        return predictions

    # multi label prediction
    def predict_multi(self,X,pcflist,labels):
        predictions = []
        for i in range(len(X)):
            f = 0
            minValues = []
            for plist in pcflist:
                values = []
                for p in plist:
                    f = np.dot(np.subtract(X[i], p.center), p.w) + p.ksi * np.linalg.norm(np.subtract(X[i], p.center),
                                                                                          ord=1) - p.gamma
                    values.append(f)
                minValues.append(np.amin(values))
            predictions.append(labels[np.argmin(minValues)])
        return predictions
