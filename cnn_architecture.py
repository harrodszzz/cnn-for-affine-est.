#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:09:38 2017

@author: yucheng
"""
from global_set import *
# settings
train_ratio = 0.8
epochs = 300
batch_size = 30
model_name = 'keras_affine_est.h5'
pretrained_model_name = 'keras_affine_est.h5'

# read data
import numpy as np
x_ori = np.load('exp_data/x_pick.npy')
x_trans = np.load('exp_data/x_trans.npy')
y = np.load('exp_data/y_trans.npy')

# split training and testing data
from math import *
length = x_ori.shape[0]
train_ind = int(floor(train_ratio * length))
# training set
tr_x_ori = x_ori[0:train_ind+1]
tr_x_trans = x_trans[0:train_ind+1]
tr_y = y[0:train_ind+1]
# testing set
te_x_ori = x_ori[train_ind+1:length]
te_x_trans = x_trans[train_ind+1:length]
te_y = y[train_ind+1:length]

# reshape images
tr_x_ori = tr_x_ori.reshape(tr_x_ori.shape[0],size,size,1)
tr_x_trans = tr_x_trans.reshape(tr_x_trans.shape[0],size,size,1)
te_x_ori = te_x_ori.reshape(te_x_ori.shape[0],size,size,1)
te_x_trans = te_x_trans.reshape(te_x_trans.shape[0],size,size,1)
# model building - import
import keras
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model

# model building - cnn feature extractions
# input:size*size images
# output: vectors of size 2048
input_img = Input(shape=(size,size,1))
conv1 = Conv2D(64,(3,3),padding='same',activation='relu')(input_img)
pool1 = MaxPooling2D((2,2))(conv1)
conv2 = Conv2D(96,(3,3),padding='same',activation='relu')(pool1)
pool2 = MaxPooling2D((2,2))(conv2)
conv3 = Conv2D(128,(3,3),padding='same',activation='relu')(pool2)
pool3 = MaxPooling2D((2,2))(conv3)
flat = Flatten()(pool3)

feature_extractor = Model(inputs=input_img,outputs=flat)

# model building - concatenate
input_1 = Input(shape=(size,size,1))
input_2 = Input(shape=(size,size,1))

feature_1 = feature_extractor(input_1)
feature_2 = feature_extractor(input_2)

concatenated = keras.layers.concatenate([feature_1,feature_2])
fc1 = Dense(1024,activation='relu')(concatenated)
out = Dense(1)(fc1)

network = Model(inputs=[input_1,input_2],outputs=out)

# load weights
import os
save_dir = os.path.join(os.getcwd(),'saved_models')
if os.path.isdir(save_dir):
    pretrained_path = os.path.join(save_dir,pretrained_model_name)
    network.load_weights(pretrained_path)

# model compile
from keras import optimizers
adam = optimizers.Adam()
import keras.backend as K
def mean_acc(y_true,y_pred):
    error = K.abs(y_true-y_pred)
    return K.mean(K.less_equal(error,3))
network.compile(optimizer=adam,loss='mse',metrics=[mean_acc])
network.fit([tr_x_ori,tr_x_trans],tr_y,epochs=epochs,batch_size=batch_size)
test_score = network.evaluate([te_x_ori,te_x_trans],te_y,batch_size=batch_size)
print("\n")
print("test loss:",test_score[0],"test acc:",test_score[1])

# model weights saving
#save_dir = os.path.join(os.getcwd(),'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir,model_name)
network.save(model_path)
print('Saved trained weights at %s'%model_path)
