# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:35:51 2020

@author: Marek
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('C:\programming\Machine Learning\Stanford course\machine-learning-ex1\machine-learning-ex1\ex1\ex1data1.txt', header = None)
data_arr = data.to_numpy()

X = np.array([data_arr[:,0]]).T
y = np.array([data_arr[:,1]]).T
m = len(y)
iterations = 1500
alpha = 0.01

plt.figure(figsize=(6,4), dpi=300)
plt.plot(X, y,'rx')

#X = np.concatenate((d, X), axis = 1)
X = np.c_[np.ones((m, 1)), X]
theta = np.zeros((2, 1))

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
    return theta

J = computeCost(X, y, theta)
J = computeCost(X, y, [[-1], [2]])

theta = (gradientDescent(X, y, theta, alpha, iterations))

predict_1 = (np.matmul(np.array([1, 3.5]), theta))*10000

plt.plot(X[:,1],np.matmul(X, theta))


