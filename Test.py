# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:09:55 2020

@author: Marek
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

h = np.array([[1],[2],[3]])
y = np.array([[1], [0], [1]])
X = np.array([[2], [2], [2]])
A = np.ones((3,1))

X = np.c_[A, X]

Z = h - y

k = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])

d = np.array([0, 2])

m = k[d,0]

kk, kkk = k.shape

fff = -k

sd = -0.1000 -12.0092 -11.2628
print(sd)

J = Z*X
sum_J = np.sum(J)

J2 = np.matmul(Z.T, X)
sum_J2 = np.sum(J2)

J3 = np.dot(Z.T, X)
sum_J3 = np.sum(J3)

# =============================================================================
# J = X*Z
# sum_J = np.sum(J)
# 
# J2 = np.matmul(X.T, Z)
# sum_J2 = np.sum(J2)
# 
# J3 = np.dot(X.T, Z)
# sum_J3 = np.sum(J3)
# 
# =============================================================================
# =============================================================================
# A = np.arange(1,19).reshape(6,3)
# B = A/3
# C = np.array([[1, 2, 3],[4, 5, 6]])
# #D = A/C
# 
# E = np.matmul(A, C.T)
# F = np.dot(A, C.T)
# #G = np.multiply(A, C.T)
# G = A*[1, 2, 3]
# 
# J = np.array([[1, 4, 9],[3, 1, 4], [2, 2, 9]])
# Jrege = np.linalg.inv(J)
# 
# K = np.matmul(J, Jrege)
# L = np.dot(J, Jrege)
# M = K*L
# =============================================================================

# A = np.ones((3, 1))
# B = np.array([[1, 2, 3],[4, 5, 6]])

# print(A.shape)
# print(B.shape)

# D = B.T

# C = (A - B)

# E = (A - D).T



#F = A*B
#G = np.dot(A,B)
# H = np.matmul(A,B)

# I = F*D
# G = np.dot(F, D)
# H = np.matmul(F,D)

# J = np.array([H[0:2, 0]])

#A = np.arange(1,10).reshape(3,3)