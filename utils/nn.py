import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Bidirectional
from tensorflow.keras.optimizers import SGD, Adam


def pad_sequence(arr, length):
    """
    This method will pad a m x n array so that m is perfectly
    divisible by length. That is, m % length == 0.
    
    Arguments:
        arr(array): m x n array where there are n input features of length m
        length(int): what we want the length of of arr to be divisible by
        
    Returns:
        padded(array): new padded array
    """
    n_features = arr.shape[1]
    remainder = arr.shape[0] % length
    if remainder == 0: # then nothing to pad
        return arr
    
    pad_length = length - remainder
    to_add = np.zeros((pad_length, n_features))
    padded = np.concatenate([arr, to_add])
    #padded = np.concatenate([to_add, arr])
    
    return padded


def batch_data(x, y, length):
    """
    Batch the neural network data, creating a shifting window of data at each time step.
    """    
    #create empty lists to append to
    X, Y = [], []
    
    #iterate over dataset, looking at moving window of 1 timestep
    #need to length to prevent out of bounds errors
    for i in range(0, len(x)-length):
        x_batch = x[i:length+i]
        y_batch = y[length+i]
        
        #append batches to lists
        X.append(x_batch)
        Y.append(y_batch)
    
    #convert lists to numpy arrays before turning into torch tensors
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    print(X.shape, Y.shape)
    return X, Y

def scale_input_data(X):
    """
    Scale input features from [-1, 1] so that it is easier to
    train a neural network.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    return scaled


def create_dnn(X, n_input, dropout, n_output, add_reg=False):
    """
    Create a DNN with or without regularization. Note: X should have
    shape (samples, timesteps, features)
    """
    timesteps = X.shape[1]
    features = X.shape[2]
    
    # design network
    dnn = tf.keras.models.Sequential()
    if(add_reg == True):
        reg = tf.keras.regularizers.l2(l=0.0001)
        dnn.add(Dense(n_input, activation='elu', kernel_regularizer=reg, input_shape=(timesteps, features)))
        dnn.add(Dropout(dropout))
        
        dnn.add(Dense(n_input, activation='elu', kernel_regularizer=reg))
        dnn.add(Dropout(dropout))
        
        dnn.add(Dense(n_input, activation='elu', kernel_regularizer=reg))
        
    else:
        dnn.add(Dense(n_input, activation='elu', input_shape=(timesteps, features)))
        dnn.add(Dropout(dropout))
        
        dnn.add(Dense(n_input, activation='elu'))
        dnn.add(Dropout(dropout))
        
        dnn.add(Dense(n_input, activation='elu'))
    
    dnn.add(Dropout(dropout))
    dnn.add(tf.keras.layers.Flatten())
    dnn.add(Dense(n_output, activation='sigmoid'))
    
    return dnn


def create_rnn(X, n_input, n_dense, dropout, n_output, add_reg=False):
    """
    Create a RNN with or without regularization. Note: X should have
    shape (samples, timesteps, features)
    """
    timesteps = X.shape[1]
    features = X.shape[2]
    
    # design network
    rnn = tf.keras.models.Sequential()
    
    if(add_reg == True):
        reg = tf.keras.regularizers.l2(l=0.0001)
        rnn.add(Bidirectional(LSTM(n_input, kernel_regularizer=reg), input_shape=(timesteps, features)))
        rnn.add(Dense(n_dense, kernel_regularizer=reg, activation='elu'))
        
    else:
        rnn.add(Bidirectional(LSTM(n_input), input_shape=(timesteps, features)))
        rnn.add(Dense(n_dense, activation='elu'))
    
    rnn.add(Dense(n_output, activation='softmax'))
    
    return rnn


def train_binary(model, X, y, X_val, y_val, loss, opt, batch, e, w):
    """
    Train a neural network with passed in hyperparameters.
    """
    # compile and fit model
    model.compile(loss=loss, 
                  optimizer=opt, 
                  metrics=[
                      tf.keras.metrics.Recall(class_id=0, name='recall_0'),
                      tf.keras.metrics.Recall(class_id=1, name='recall_1'),
                      tf.keras.metrics.Precision(class_id=0, name='precision_0'),
                      tf.keras.metrics.Precision(class_id=1, name='precision_1'),
    ])

    history = model.fit(X, y, validation_data=(X_val, y_val),
                        batch_size=batch, epochs=e, verbose=0, 
                        shuffle=False, class_weight=w)
    
    return history


def train_multiclass(model, X, y, X_val, y_val, loss, opt, batch, e, w):
    """
    Train a neural network with passed in hyperparameters.
    """
    # compile and fit model
    model.compile(loss=loss, 
                  optimizer=opt, 
                  metrics=[
                      tf.keras.metrics.Recall(class_id=0, name='recall_0'),
                      tf.keras.metrics.Recall(class_id=1, name='recall_1'),
                      tf.keras.metrics.Recall(class_id=2, name='recall_2'),
                      tf.keras.metrics.Precision(class_id=0, name='precision_0'),
                      tf.keras.metrics.Precision(class_id=1, name='precision_1'),
                      tf.keras.metrics.Precision(class_id=2, name='precision_2'),
    ])

    history = model.fit(X, y, validation_data=(X_val, y_val),
                        batch_size=batch, epochs=e, verbose=0, 
                        shuffle=False, class_weight=w)
    
    return history


def train_problems(model, X, y, X_val, y_val, loss, opt, batch, e, w):
    """
    Train a neural network with passed in hyperparameters.
    """
    # compile and fit model
    model.compile(loss=loss, 
                  optimizer=opt, 
                  metrics=[
                      tf.keras.metrics.Recall(class_id=0, name='recall_0'),
                      tf.keras.metrics.Recall(class_id=1, name='recall_1'),
                      tf.keras.metrics.Recall(class_id=2, name='recall_2'),
                      tf.keras.metrics.Recall(class_id=3, name='recall_3'),
                      tf.keras.metrics.Recall(class_id=4, name='recall_4'),
                      tf.keras.metrics.Precision(class_id=0, name='precision_0'),
                      tf.keras.metrics.Precision(class_id=1, name='precision_1'),
                      tf.keras.metrics.Precision(class_id=2, name='precision_2'),
                      tf.keras.metrics.Precision(class_id=3, name='precision_3'),
                      tf.keras.metrics.Precision(class_id=4, name='precision_4'),
    ])

    history = model.fit(X, y, validation_data=(X_val, y_val),
                        batch_size=batch, epochs=e, verbose=0, 
                        shuffle=False, class_weight=w)
    
    return history