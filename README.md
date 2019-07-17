# Revised-Polyhedral-Conic-Function Algorithm for Supervised Classification
This repository contains, implementation of the polyhedral conic function algorithms and experiment setups. If you use these implementation in a scientific publication, we would appreciate citations to the following paper:



Requirments:
      Python 2.7
      Numpy
      Scikit-Learn
      Gurobi Optimization Tool, gurobi.py

Implemented Algorithms:

        PCF.py:         Implementation of the algorithm presented in 
                        Seperation via polyhedral conic functions, Gasimov and Ozturk, 2006

       kmeansPCF.py:    Implementation of kmeans PCF Algorithm presented in
                        Journal of Industrial & Management Optimization 11.3 (2015): 921-932
       
       rPCF.py:        Implementation of Revised Polyhedral Conic Functions Algorithm presented in


Experiment Setups:

     experimentSetupPCF.py:               This file contains experiment setup  for PCF Algorithm by using PCF.py
                                                -classification, binary 
                                                -sampling, srattified 10-fold cross validation
                                   
     experimentSetupPCFMultiLabel.py:     This file contains experiment setup  for PCF Algorithm by using PCF.py
                                                -classification, multiclass, one vs All
                                                -sampling, training-test
                                             
     experimentSetupKmeansPCF.py:         This file contains experiment setup  for PCF Algorithm by using kmeansPCF.py
                                                -classification, multiclass, one vs All
                                                -sampling, training-test
                                        
     experimentSetupRPCF.py:              This file contains experiment setup  for revised PCF Algorithm by using rPCF.py
                                                -classification, binary 
                                                -sampling, srattified 10-fold cross validation
                             
     experimentSetupRPCFMultiLabel.py:    This file contains experiment setup  for revised PCF Algorithm by using rPCF.py
                                                -classification, multiclass, one vs All
                                                -sampling, training-test
