# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 21:50:16 2020

@author: Marek
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sc
import scipy.io as sio

data = sio.loadmat('C:\programming\Machine Learning\Stanford course\machine-learning-ex3\machine-learning-ex3\ex3\ex3data1.mat')

X = data['X']
y = data['y']

input_layer_size = 400
num_labels = 10

# sel_idx = random.sample(range(m), 100)
# sel = X[sel_idx,:]

theta_t = np.array([[-2, -1, 1, 2]]).T
X_t = np.c_[np.ones((5, 1)), np.arange(1, 16).reshape(3, 5).T/10]
y_t = np.array([[1, 0, 1, 0, 1]]).T
lambda_t = 3
lamb = 0.1

def costFunctionReg(theta, X, y, lamb):
    m = X.shape[0]
    theta = theta.reshape(-1, 1)
    h = np.array(1/(1 + np.exp((-1)*np.matmul(X, theta))))
    J = 1/m*np.sum(np.matmul(-y.T, np.log(h)) - np.matmul((1-y).T,np.log(1-h))) + lamb/(2*m)*np.sum(theta[1:,:]**2)
    
    grad_0 = 1/m*(np.matmul(X[:,0].T,(h-y)))
    grad_rest = 1/m*(np.matmul(X[:,1:].T,(h-y))) + (np.array([lamb/m*theta[1:,0]])).T
    grad = np.vstack((grad_0, grad_rest))

    return J, grad

def OneVsAll(X, y, num_labels, lamb):
    
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    initial_theta = np.zeros((n + 1, 1))
    
    all_theta = np.empty((401, 0))
    
    for i in range (1, num_labels + 1):
        y_bin = np.zeros((m, 1))
        for j, item in enumerate(y):
            if i == item:
                y_bin[j, 0] = 1
        result = sc.fmin_tnc(func=costFunctionReg, x0=initial_theta, args=(X, y_bin, lamb))
        theta  = np.array([result[0]]).T
        all_theta = np.append(all_theta, theta, axis = 1)
    return all_theta

def predict(theta, X, y):
    
    m, n = X.shape
    X = np.c_[np.ones((m, 1)), X]
    p = np.zeros((m,1))
    h = np.array(1/(1 + np.exp((-1)*np.matmul(X, theta))))
    p = np.argmax(h, axis = 1)+1
    acc = (np.sum([1 for i in range(len(p)) if p[i] == y[i,0]]))/m
    return acc, p

J, grad  = costFunctionReg(theta_t, X_t, y_t, lambda_t)
all_theta = OneVsAll(X, y, num_labels, lamb)
acc, p = predict (all_theta, X, y)