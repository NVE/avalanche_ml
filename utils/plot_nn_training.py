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


def plot_history(history):
    """
    Plot the loss function vs epochs and metric vs epochs after
    training a neural network.
    """
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))

    # plot loss over time
    ax[0, 0].plot(history['loss'], label='train')
    ax[0, 0].plot(history['val_loss'], label='val')
    
    # recall plot
    ax[0, 1].plot(history['recall_0'], color='tab:blue', alpha=0.5, label='Train_0')
    ax[0, 1].plot(history['val_recall_0'], color='tab:blue', label='Val_0')
    ax[0, 1].plot(history['recall_1'], color='tab:orange', alpha=0.5, label='Train_1')
    ax[0, 1].plot(history['val_recall_1'], color='tab:orange', label='Val_1')
    
    # precision plot
    ax[1, 1].plot(history['precision_0'], color='tab:blue', alpha=0.5, label='Train_0')
    ax[1, 1].plot(history['val_precision_0'], color='tab:blue', label='Val_0')
    ax[1, 1].plot(history['precision_1'], color='tab:orange', alpha=0.5, label='Train_1')
    ax[1, 1].plot(history['val_precision_1'], color='tab:orange', label='Val_1')

    # add titles and legends
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

    # add titles and legends
    ax[1, 0].set_axis_off()

    ax[0, 0].set_title('Loss')
    ax[0, 1].set_title('Recall')
    ax[1, 1].set_title('Precision')

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 1].legend()
    plt.show()
    

def plot_history_problems(history):
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
    ax[0, 1].plot(history['recall_3'], color='tab:purple', alpha=0.5, label='Train_3')
    ax[0, 1].plot(history['val_recall_3'], color='tab:purple', label='Val_3')
    ax[0, 1].plot(history['recall_4'], color='tab:gray', alpha=0.5, label='Train_4')
    ax[0, 1].plot(history['val_recall_4'], color='tab:gray', label='Val_4')

    # precision plot
    ax[1, 1].plot(history['precision_0'], color='tab:blue', alpha=0.5, label='Train_0')
    ax[1, 1].plot(history['val_precision_0'], color='tab:blue', label='Val_0')
    ax[1, 1].plot(history['precision_1'], color='tab:orange', alpha=0.5, label='Train_1')
    ax[1, 1].plot(history['val_precision_1'], color='tab:orange', label='Val_1')
    ax[1, 1].plot(history['precision_2'], color='tab:green', alpha=0.5, label='Train_2')
    ax[1, 1].plot(history['val_precision_2'], color='tab:green', label='Val_2')
    ax[1, 1].plot(history['precision_3'], color='tab:purple', alpha=0.5, label='Train_3')
    ax[1, 1].plot(history['val_precision_3'], color='tab:purple', label='Val_3')
    ax[1, 1].plot(history['precision_4'], color='tab:gray', alpha=0.5, label='Train_4')
    ax[1, 1].plot(history['val_precision_4'], color='tab:gray', label='Val_4')

    # add titles and legends
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