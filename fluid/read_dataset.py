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
def data_from_name(name, noise = 0.0, theta=2.4):     
    if name == 'fluid':
        return fluid()    
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


def fluid():
    

    data = pd.read_csv("FluidFlowBox_traj1_x.csv")
    #data = pd.read_csv("FluidFlowBox_traj3_x.csv")

    noise=0.001*0.001 #noise=0.001*0.005

    
    X = np.array(data)
    Xclean = X.copy()
    
    X[:,0] += np.random.standard_normal(X[:,0].shape) * noise
    X[:,1] += np.random.standard_normal(X[:,1].shape) * noise
    X[:,2] += np.random.standard_normal(X[:,2].shape) * noise
    
    # Rotate to high-dimensional space
    Q = np.random.standard_normal((96,3))
    Q,_ = np.linalg.qr(Q)
    
    X = X.dot(Q.T) # rotate
    Xclean = Xclean.dot(Q.T)


    # scale 
    X = 2 * (X - np.min(X)) / np.ptp(X) - 1
    Xclean = 2 * (Xclean - np.min(Xclean)) / np.ptp(Xclean) - 1

    
    # split into train and test set 
    X_train = X[0:800]   
    X_test = X[800:1730]

    X_train_clean = Xclean[0:800]   
    X_test_clean = Xclean[800:1730]     
    
    #******************************************************************************
    # Return train and test set
    #******************************************************************************
    return X_train, X_test, X_train_clean, X_test_clean, 96, 1
