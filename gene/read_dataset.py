import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from matplotlib import pylab as plt
from scipy.special import ellipj, ellipk

import torch
import pandas as pd

#******************************************************************************
# Read in data
#******************************************************************************
def data_from_name(name, noise = 0.0, theta=0.8):     
    if name == 'gene':
        return gene()  
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


def gene():
    
    X = pd.read_csv('gene.csv')
    print(X.shape)  #84维 22步 （84，22）
    
    X = X.T
    X=np.array(X)
    print(X.shape)
    print(type(X))
    
    xmin=np.min(X)
    xptp=np.ptp(X)
    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    print(X.shape)

    
    # split into train and test set 
    X_train = X[0:15]   
    X_test = X[15:]    
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test,84, 1,xmin,xptp


gene()