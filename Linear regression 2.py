# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:20:49 2020

@author: Marek
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def computeCost(X, y, theta):
    
    h = np.matmul(X, theta)
    J = 1/(2*m)*np.sum((h - y)**2)

    return J

def gradientDescent (X, y, theta, alpha, num_itr):
    
    J_history = np.zeros((num_itr, 1))
    
    for itr in range(num_itr):
        
        h = np.matmul(X, theta)
        theta = theta - alpha/m * ((np.dot((h - y).T, X)).T)
        J_history[itr] = computeCost(X, y, theta)
    
    return theta, J_history

def featureNormalize(X_norm):
    
    mu = np.mean(X_norm, axis = 0)
    sigma = np.std(X_norm, axis = 0)
    X_norm = (X - mu)/sigma
    
    
    
    return X_norm

def normalEqn(X, y):

    theta_1 = np.linalg.inv(np.matmul(X.T, X))
    theta = np.matmul(np.matmul(theta_1, X.T), y)
    return theta

data = pd.read_csv('C:\programming\Machine Learning\Stanford course\machine-learning-ex1\machine-learning-ex1\ex1\ex1data2.txt', header = None)
data_arr = data.to_numpy()

X = np.array(data_arr[:,:2])
y = np.array([data_arr[:,2]]).T
m = len(y)
iterations = 400
alpha = 0.1


X = featureNormalize(X)
X = np.c_[np.ones((m, 1)), X]

theta = np.zeros((3, 1))
(theta, J_history) = gradientDescent(X, y, theta, alpha, iterations)

plt.figure(figsize=(6, 4), dpi=300)
plt.plot(J_history,'r-')
plt.xlabel('Iterations')
plt.ylabel('Cost J')


prediction_1 = np.matmul([1, 1650, 3], theta)

X_n = np.array(data_arr[:,:2])
X_n = np.c_[np.ones((m, 1)), X_n]
y_n = np.array([data_arr[:,2]]).T

theta_norm_eqn = normalEqn(X_n, y_n)
prediction_2 = np.matmul([1, 1650, 3], theta_norm_eqn)
