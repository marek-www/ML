# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 23:04:23 2020

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

    plt.plot(X[idx_0,1], X[idx_0,2],"ko", markersize = 7, markerfacecolor = 'yellow')
    plt.plot(X[idx_1,1], X[idx_1,2],"k+", markersize = 7, linewidth = 2)

def costFunction(theta, X, y):
    theta = theta.reshape(-1, 1)
    h = np.array(1/(1 + np.exp((-1)*np.matmul(X, theta))))
    J = 1/m*np.sum(np.matmul(-y.T, np.log(h)) - np.matmul((1-y).T,np.log(1-h)))
    grad = 1/m*(np.matmul(X.T,(h-y)))
    return J, grad

def plotBoundary(theta, X, y):
    
    plot_X1 = [np.amax(X[:,1]), np.amin(X[:,1])]
    plot_X2 = (-1/theta[2])*(theta[1]*plot_X1 + theta[0])
    plt.plot(plot_X1,plot_X2)

def sigmoid(z):
    
    g = np.array(1/(1 + np.exp((-1)*z)))
    return g

def predict(theta, X, y):
    
    p = np.zeros((m,1))
    h = np.array(1/(1 + np.exp((-1)*np.matmul(X, theta))))
    idx_1 = [i for i in range(len(h)) if h[i,0] >= 0.5]
    p[idx_1,0] = 1
    acc = (np.sum([1 for i in range(len(p)) if p[i,0] == y[i,0]]))/m
    return acc

plt.figure(figsize=(6,4), dpi=300)
plotData(X, y)
cost, grad = costFunction(initial_theta, X, y)

test_theta = np.array([[-24, 0.2, 0.2]]).T
cost, grad = costFunction(test_theta, X, y)

result = sc.fmin_tnc(func=costFunction, x0=initial_theta, args=(X, y))
theta  = np.array([result[0]]).T
plotBoundary(theta, X, y)

test_prob = sigmoid(np.dot((np.array([1, 45, 85])), theta))
accuracy = predict(theta, X, y)



