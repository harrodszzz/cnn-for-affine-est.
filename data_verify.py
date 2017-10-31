#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:01:06 2017

@author: yucheng
"""

import numpy as np
x_ori = np.load('exp_data/x_pick.npy')
x_trans = np.load('exp_data/x_trans.npy')
y = np.load('exp_data/y_trans.npy')

import matplotlib.pyplot as plt
pic_ind = 0
for i in range(1000,1010):
    plt.figure(pic_ind)
    plt.subplot(121)
    plt.imshow(x_ori[i])
    plt.title('origin')
    plt.subplot(122)
    plt.imshow(x_trans[i])
    plt.title(y[i])
    pic_ind += 1