# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 20:41:03 2020

@author: Marek
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sc

data = pd.read_csv('C:\programming\Machine Learning\Stanford course\machine-learning-ex2\machine-learning-ex2\ex2\ex2data2.txt', header = None)
data_arr = data.to_numpy()

X = np.array(data_arr[:,:2])
y = np.array([data_arr[:,2]]).T
m, n = X.shape
lamb = 1

def plotData(X, y):
    
    idx_0 = [i for i in range(len(y)) if y[i,0] == 0]
    idx_1 = [i for i in range(len(y)) if y[i,0] == 1]

    plt.plot(X[idx_0,0], X[idx_0,1],"ko", markersize = 7, markerfacecolor = 'yellow', label = 'y = 0')
    plt.plot(X[idx_1,0], X[idx_1,1],"k+", markersize = 7, linewidth = 2, label = 'y = 1')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()

def mapFeature(X1, X2):
    
    degree = 6
    terms = np.ones(X1.shape)
    col = np.zeros(X1.shape)
    
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            col = (X1**(i-j))*(X2**(j))
            terms = np.vstack((terms, col))
    return terms.T
    
def costFunctionReg(theta, X, y, lamb):
    theta = theta.reshape(-1, 1)
    h = np.array(1/(1 + np.exp((-1)*np.matmul(X, theta))))
    J = 1/m*np.sum(np.matmul(-y.T, np.log(h)) - np.matmul((1-y).T,np.log(1-h))) + lamb/(2*m)*np.sum(theta[1:,:]**2)
    
    grad_0 = 1/m*(np.matmul(X[:,0].T,(h-y)))
    grad_rest = 1/m*(np.matmul(X[:,1:].T,(h-y))) + (np.array([lamb/m*theta[1:,0]])).T
    grad = np.vstack((grad_0, grad_rest))

    return J, grad

def predict(theta, X, y):
    
    p = np.zeros((m,1))
    h = np.array(1/(1 + np.exp((-1)*np.matmul(X, theta))))
    idx_1 = [i for i in range(len(h)) if h[i,0] >= 0.5]
    p[idx_1,0] = 1
    acc = (np.sum([1 for i in range(len(p)) if p[i,0] == y[i,0]]))/m
    return acc

def plotBoundary(theta, X, y):
    
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u),len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(mapFeature(u[i],v[j]),theta)

    z = z.T
    plt.contour(u, v, z, [0])
    
plt.figure(figsize=(6,4), dpi=300)
plotData(X, y)

X = mapFeature(X[:, 0],X[:, 1])
initial_theta = np.zeros((np.size(X, axis=1), 1))
test_theta = np.ones((np.size(X, axis=1), 1))

J, grad = costFunctionReg(initial_theta, X, y, lamb)
J, grad = costFunctionReg(test_theta, X, y, 10)

result = sc.fmin_tnc(func=costFunctionReg, x0=initial_theta, args=(X, y, lamb))
theta  = np.array([result[0]]).T

accuracy = predict(theta, X, y)
plotBoundary(theta, X, y)