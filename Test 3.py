# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 00:15:36 2020

@author: Marek
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sc

data = pd.read_csv('C:\programming\Machine Learning\Stanford course\machine-learning-ex2\machine-learning-ex2\ex2\ex2data1.txt', header = None)
data_arr = data.to_numpy()

X = np.array(data_arr[:,:2])
y = np.array([data_arr[:,2]]).T
m, n = X.shape
X = np.c_[np.ones((m, 1)), X]
initial_theta = np.zeros((n + 1, 1))

def plotData(X, y):
    
    idx_0 = [i for i in range(len(y)) if y[i,0] == 0]
    idx_1 = [i for i in range(len(y)) if y[i,0] == 1]

    plt.plot(X[idx_0,0], X[idx_0,1],"ko", markersize = 7, markerfacecolor = 'yellow')
    plt.plot(X[idx_1,0], X[idx_1,1],"k+", markersize = 7, linewidth = 2)

def costFunction(theta, X, y):
    theta = theta.reshape(-1, 1)
    h = np.array(1/(1 + np.exp((-1)*np.matmul(X, theta))))
    J = 1/m*np.sum(np.matmul(-y.T, np.log(h)) - np.matmul((1-y).T,np.log(1-h)))
    #grad = 1/m*(np.matmul(X.T,(h-y)))
    return J

def gradient1(theta, X, y):
    theta = theta.reshape(-1, 1)
    h = np.array(1/(1 + np.exp((-1)*np.matmul(X, theta))))
    #J = 1/m*np.sum(np.matmul(-y.T, np.log(h)) - np.matmul((1-y).T,np.log(1-h)))
    grad = 1/m*(np.matmul(X.T,(h-y)))
    return np.ndarray.flatten(grad)

plt.figure(figsize=(6,4), dpi=300)
plotData(X, y)
cost= costFunction(initial_theta, X, y)
grad = gradient1(initial_theta, X, y)

test_theta = np.array([[-24, 0.2, 0.2]]).T
cost= costFunction(test_theta, X, y)
grad = gradient1(test_theta, X, y)

result = sc.fmin_bfgs(costFunction, initial_theta, fprime=gradient1, args=(X, y))
