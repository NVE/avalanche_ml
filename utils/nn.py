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


def train(model, X, y, X_val, y_val, loss, opt, batch, e, w):
    """
    Train a neural network with passed in hyperparameters.
    """
    # compile and fit model
    model.compile(loss=loss, 
                  optimizer=opt, 
                  metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

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


def plot_history(history):
    """
    Plot the loss function vs epochs and metric vs epochs after
    training a neural network.
    """
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))

    # plot loss over time
    ax[0, 0].plot(history['loss'], label='train')
    ax[0, 0].plot(history['val_loss'], label='val')
    
    # plot metrics over time
    ax[0, 1].plot(history['recall'], label='train')
    ax[0, 1].plot(history['val_recall'], label='val')
    
    ax[1, 1].plot(history['precision'], label='train')
    ax[1, 1].plot(history['val_precision'], label='val')
    
    ax[0, 0].set_title('Loss')
    ax[0, 1].set_title('Recall')
    ax[1, 1].set_title('Precision')
    
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 1].legend()
    
    ax[1, 0].set_axis_off()
    
    plt.show()
    

def plot_history_multiclass(history):
    """
    Plot the loss function vs epochs and metric vs epochs after
    training a neural network.
    """
    fig, ax = plt.subplots(2, 2, figsize=(14, 14))

    # loss plot
    ax[0, 0].plot(history['loss'], label='Train')
    ax[0, 0].plot(history['val_loss'], label='Val')

    # recall plot
    ax[0, 1].plot(history['recall_0'], color='tab:blue', alpha=0.5, label='Train_0')
    ax[0, 1].plot(history['val_recall_0'], color='tab:blue', label='Val_0')
    ax[0, 1].plot(history['recall_1'], color='tab:orange', alpha=0.5, label='Train_1')
    ax[0, 1].plot(history['val_recall_1'], color='tab:orange', label='Val_1')
    ax[0, 1].plot(history['recall_2'], color='tab:green', alpha=0.5, label='Train_2')
    ax[0, 1].plot(history['val_recall_2'], color='tab:green', label='Val_2')

    # precision plot
    ax[1, 1].plot(history['precision_0'], color='tab:blue', alpha=0.5, label='Train_0')
    ax[1, 1].plot(history['val_precision_0'], color='tab:blue', label='Val_0')
    ax[1, 1].plot(history['precision_1'], color='tab:orange', alpha=0.5, label='Train_1')
    ax[1, 1].plot(history['val_precision_1'], color='tab:orange', label='Val_1')
    ax[1, 1].plot(history['precision_2'], color='tab:green', alpha=0.5, label='Train_2')
    ax[1, 1].plot(history['val_precision_2'], color='tab:green', label='Val_2')

    # extra details
    ax[1, 0].set_axis_off()
    ax[0, 0].set_title('Loss')
    ax[0, 1].set_title('Recall')
    ax[1, 1].set_title('Precision')
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 1].legend()
    plt.show()
    
    
def plot_confusion_matrix(model, X, y_true):
    y_pred = np.argmax(model.predict(X), axis=1)
    precision, recall, fscore, support = score(y_true, y_pred)
    print('precision: {}'.format(precision))
    print('recall:    {}'.format(recall))
    print('fscore:    {}'.format(fscore))

    #use seaborn's sns.heatmap() function for pretty plotting of confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='d', cbar=False)

    #set x and y labels, as well as title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('ROC Confusion Matrix')

    plt.show()
    
    return y_pred