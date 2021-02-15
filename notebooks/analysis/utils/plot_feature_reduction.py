import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd


def plot_importances(df, labels, rf):
    """
    Input a dataframe as well as trained RandomForest model, plot feature importances.
    
    """
    columns = df.columns # all columns except the target column
    importances_rf = rf.best_estimator_.feature_importances_
    
    #take the top x feature importances to plot
    idx_all = np.argsort(importances_rf)
    idx_rf = np.argsort(importances_rf)[-30:]
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    #random forest and xgboost feature importances on the same plot
    ax.barh(range(len(idx_rf)), importances_rf[idx_rf], color='tab:blue', align='center', alpha=0.5, label='RF')
    
    ax.set_yticks(range(len(idx_rf)))
    ax.set_yticklabels([columns[i] for i in idx_rf], fontsize=10)
    ax.set_xlabel('Relative Importance')
    ax.set_title('Largest Feature Importances')
    ax.legend()

    plt.show()
    
    #return a list of columns sorted by feature importance
    return importances_rf[idx_all][::-1], [columns[i] for i in idx_all][::-1]


def plot_explained_variance(variance, cum_variance, idx90, idx95):
    """
    Plot the ratio of explained variance per principal component, as well
    as the cumulative explained variance across all principal components.
    
    Arguments:
        variance(np.array): pca.explained_variance_ratio from scikit-learn
        cum_variance(np.array): np.cumsum(pca.explained_variance_ratio) from scikit-learn
        idx90(int): location in cum_variance where 90% explained variance is reached
        idx95(int): location in cum_variance where 95% explained variance is reached
    """
    # investigate the variance accounted for by each principal component.
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))

    ax[0].plot(variance)
    ax[1].plot(cum_variance)

    ax[0].set_xlabel('Principal Component #')
    ax[0].set_ylabel('Ratio of Explained Variance')
    ax[0].set_title('Ratio of Explained Variance Per Principal Component')

    ax[1].set_xlabel('Principal Component #')
    ax[1].set_ylabel('Cumulative Explained Variance')
    ax[1].set_title('Cumulative Explained Variance Across Principal Components')

    # show where the number of pcs crosses a certain level of cumulative variance
    ax[1].axhline(y=0.90, linestyle='--', color='r')
    ax[1].axvline(x=idx90, linestyle='--', color='r')

    ax[1].axhline(y=0.95, linestyle='--', color='g')
    ax[1].axvline(x=idx95, linestyle='--', color='g')

    plt.show()
    

def plot_data_clusters(x2, x3, colors, title):
    """
    Plot a 2D and 3D t-SNE matrix according to the values given by colors. Colors is a 1D
    array of values from our labels using the column title.
    
    Arguments:
        x2(np.array): 2D t-SNE embedding
        x3(np.array): 3D t-SNE embedding
    """    
    # 0 -> len(y.columns), encoding column names to numbers
    unique_classes = np.unique(colors)
    num_classes = len(unique_classes)
    class_indices = [np.where(unique_classes==val)[0][0] for val in colors.astype(np.int)]

    # choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_classes))

    #### 2D plotting ####
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(121, aspect='equal')
    sc = ax.scatter(x2[:, 0], x2[:, 1], lw=0, s=40, c=palette[class_indices])

    # add the labels for each digit corresponding to the label
    txts_2d = []
    for i in range(num_classes):
        # position of each label at median of data points.
        xtext, ytext = np.mean(x2[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(unique_classes[i]), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts_2d.append(txt)
    
    #### 3D plotting ####
    ax3d = fig.add_subplot(122, projection='3d')
    ax3d.scatter(*zip(*x3), alpha=0.7, c=palette[class_indices])

    # add the labels for each digit corresponding to the label
    txts_3d = []
    for i in range(num_classes):
        # position of each label at median of data points.
        xtext, ytext, ztext = np.median(x3[colors == i, :], axis=0)
        txt = ax3d.text(xtext, ytext, ztext, str(unique_classes[i]), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts_3d.append(txt)

    fig.tight_layout()
    plt.suptitle(title, y=0.92, fontsize=16)
    plt.show()