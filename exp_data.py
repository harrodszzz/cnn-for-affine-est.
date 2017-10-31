#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:28:56 2017

@author: yucheng
"""
# Run before every experiment, prepare the experimental data
from global_set import *
# parameters settings
pick_up = [4] # which digits to pick up 

# read data
import numpy as np
x = np.load('x.npy')
y = np.load('y.npy')

# pick up the digits
x_pick = x[y==pick_up]
y_pick = y[y==pick_up]

# resize images
import cv2
x_resize = np.zeros([x_pick.shape[0],size,size])
for index in range(0,x_pick.shape[0]):
    x_res = cv2.resize(x_pick[index],(size,size),interpolation=cv2.INTER_CUBIC)
    x_resize[index] = x_res
    
# affine transform
from math import *
x_transform = np.zeros([x_pick.shape[0],size,size])
y_transform = np.zeros([x_pick.shape[0]])
for index in range(0,x_pick.shape[0]):
    rotation_degrees = 180 * np.random.random() + np.random.normal()
    M = cv2.getRotationMatrix2D((size/2,size/2),rotation_degrees,1)
    x_trans = cv2.warpAffine(x_resize[index],M,(size,size))
    x_transform[index] = x_trans
    y_transform[index] = rotation_degrees

# save data
import os
path = os.path.join(os.getcwd(),'exp_data')
if not os.path.isdir(path):
    os.makedirs(path)
np.save('exp_data/x_pick',x_resize)
np.save('exp_data/x_trans',x_transform)
np.save('exp_data/y_trans',y_transform)