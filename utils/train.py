import os
import pickle
import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

def train_forest(X, y, kind):
    """
    Train a random forest regression model on input data X and labels y, then
    return the model for further processing.
    
    Arguments:
        X(DataFrame): input data
        y(DataFrame): labels
        kind(str): what type of labels you are training on
        
    Returns:
        forest_model(RandomForestRegressor): trained random forest models after cross-fold validation
    """
    # makes sure the models and grid search object give the same results every time
    random = 0
    num_iter = 10

    #instantiate models here
    forest = RandomForestClassifier(random_state=random)

    #define parameters dictionaries here
    ensemble_params = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
                 'max_depth': [1, 2, 4, 6, 8, 10, 20, 30, None],  #higher values often lead to overfitting
                 'max_features': ['auto', 'sqrt', 'log2'],
                 'min_samples_leaf': [1, 2, 4],
                 'min_samples_split': [2, 5, 10],
                'bootstrap': [True, False]
                }

    #create our random search cv objects
    forest_search = RandomizedSearchCV(estimator=forest,
                            param_distributions=ensemble_params,
                            n_iter=num_iter,         #number of parameter settings sampled
                            scoring='neg_root_mean_squared_error',
                            n_jobs=-1,         #use all processors available
                            cv=5,
                            verbose=2,         #print results during tuning
                            random_state=random
    )

    #now fit the search objects and return them
    print('Training model on {} label(s)...'.format(kind))
    forest_model = forest_search.fit(X, y)
    
    #save model to ../data folder when done
    print('Saving model...')
    directory = os.path.join(os.getcwd(), 'models')
    filename = 'model_{}.sav'.format(kind)
    path = os.path.join(directory, filename)
    
    with open(path, 'wb') as file:
        pickle.dump(forest_model, file)
    
    print()
    
    return forest_model

