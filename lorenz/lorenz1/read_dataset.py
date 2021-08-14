import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl

import torch
import pandas as pd

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, noise = 0.0, theta=2.4):    
    if name == 'lorenz':
        return lorenz()    
    else:
        raise ValueError('dataset {} not recognized'.format(name))


def rescale(Xsmall, Xsmall_test):
    #******************************************************************************
    # Rescale data
    #******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()
    
    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin)) 
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin)) 

    return Xsmall, Xsmall_test


def lorenz():
    xs, ys, zs = [], [], []

    a, b, c = 10.0, 28.0, 8.0 / 3.0         
    h = 0.006                            
    x0, y0, z0 = 0.1, 0, 0
    for i in range(5000):
        x1 = x0 + h * a * (y0 - x0)
        y1 = y0 + h * (x0 * (b - z0) - y0)
        z1 = z0 + h * (x0 * y0 - c * z0)
        x0, y0, z0 = x1, y1, z1
        xs.append(x0)
        ys.append(y0)
        zs.append(z0)
    X=[]
    X.append(xs)
    X.append(ys)
    X.append(zs)
    X=np.array(X).T
    Xclean=X.copy()

    # Rotate to high-dimensional space
    Q = np.random.standard_normal((96,3))
    Q,_ = np.linalg.qr(Q)
    
    X = X.dot(Q.T) # rotate
    Xclean = Xclean.dot(Q.T)
    

    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1
    train_num = 1000

    X_train = X[:train_num]   
    X_test = X[train_num:]
    X_train_clean = Xclean[:train_num]   
    X_test_clean = Xclean[train_num:]


    return X_train, X_test, X_train_clean, X_test_clean, 96, 1
