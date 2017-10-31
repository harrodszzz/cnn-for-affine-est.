#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:15:57 2017

@author: yucheng
"""
# 1st script to run, read in some data


# download mnist data
from keras.datasets import mnist
(x_tr,y_tr),(x_te,y_te) = mnist.load_data()

# concatenate train and test 
import numpy as np
x = np.concatenate((x_tr,x_te),axis=0)
y = np.concatenate((y_tr,y_te),axis=0)

# save data
np.save('x',x)
np.save('y',y)