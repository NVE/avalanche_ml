import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from avaml.aggregatedata import ForecastDataset, LabeledData

def preprocess(ld):
    """
    This method names hierarchical indeces, drops region columns, replaces dashes with underscores,
    and processes several columns so that they are type int instead of type str.
    
    Arguments:
        ld(LabeledDataset): input data set read from CSV or downloaded from remote source
        
    Returns:
        ld(LabeledDataset): processed data set
    """
    # first, drop regions
    ld.data = ld.drop_regions().data

    # then, rename indices
    ld.data = ld.data.rename_axis(['date','region'])
    ld.label = ld.label.rename_axis(['date', 'region'])

    # flatten the hierchy of columns to 1D
    ld.data.columns = [' '.join(col).strip().replace(' ', '_') for col in ld.data.columns.values]
    ld.label.columns = [' '.join(col).strip().replace(' ', '_') for col in ld.label.columns.values]

    # replace double underscores with single underscores
    ld.data.columns = [col.replace('__', '_') for col in ld.data.columns.values]
    ld.label.columns = [col.replace('__', '_') for col in ld.label.columns.values]

    # replace dashes with single underscores
    ld.data.columns = [col.replace('-', '_') for col in ld.data.columns.values]
    ld.label.columns = [col.replace('-', '_') for col in ld.label.columns.values]

    # convert some columns in labels to type int for averaging
    ld.label['CLASS_problem_amount'] = ld.label['CLASS_problem_amount'].astype(int)
    ld.label['CLASS_danger_level'] = ld.label['CLASS_danger_level'].astype(int)

    for column in ld.label.columns:
        if column.endswith(('_dist', '_dsize', '_lev_fill', '_prob', '_trig')):
            ld.label[column] = ld.label[column].astype(int)

    # below, we can try to make categorical variables in the labels numeric
    # first for the emergency warning column
    warning_dict = {'Ikke gitt':0,
                    'Naturlig utløste skred':1}

    ld.label['CLASS_emergency_warning'] = ld.label['CLASS_emergency_warning'].replace(warning_dict)

    # and now for the class problems
    problem1 = list(np.unique(ld.label.loc[:, 'CLASS_problem_1'].values))
    problem2 = list(np.unique(ld.label.loc[:, 'CLASS_problem_2'].values))
    problem3 = list(np.unique(ld.label.loc[:, 'CLASS_problem_3'].values))

    list_of_problems = sorted(list(np.unique(problem1 + problem2 + problem3)))
    problems_dict = {'':0, 'drift-slab':1, 'glide':2, 'new-loose':3,
                     'new-slab':4, 'pwl-slab':5, 'wet-loose':6, 'wet-slab':7}

    ld.label['CLASS_problem_1'] = ld.label['CLASS_problem_1'].replace(problems_dict)
    ld.label['CLASS_problem_2'] = ld.label['CLASS_problem_2'].replace(problems_dict)
    ld.label['CLASS_problem_3'] = ld.label['CLASS_problem_3'].replace(problems_dict)
    
    return ld


def encode_causes(l):
    """
    One-hot encode the avalanche cause columns.
    
    Arguments:
        l(DataFrame): labels for data set prior to one-hot encoding
        
    Returns:
        l(DataFrame): labels for data set after to one-hot encoding
    """
    cause_columns = ['CLASS_drift_slab_cause', 'CLASS_glide_cause', 'CLASS_new_loose_cause',
                     'CLASS_new_slab_cause', 'CLASS_pwl_slab_cause', 'CLASS_wet_loose_cause', 
                     'CLASS_wet_slab_cause']
    l = pd.get_dummies(l, columns=cause_columns, dtype=np.int64)
    l.columns = [col.replace('-', '_') for col in l.columns.values]
    
    return l


def encode_aspects(l):
    """
    Encode the avalanche aspect columns in the labels of our data set.
    
    Arguments:
        l(DataFrame): labels for data set prior to encoding the aspect columns
        
    Returns:
        l(DataFrame): labels for data set after to encoding the aspect columns
    """
    aspect_columns = ['MULTI_drift_slab_aspect', 'MULTI_glide_aspect', 'MULTI_new_loose_aspect', 
                      'MULTI_new_slab_aspect', 'MULTI_pwl_slab_aspect', 'MULTI_wet_loose_aspect', 
                      'MULTI_wet_slab_aspect']
    
    for column in aspect_columns:
        for i in range(0, 8):
            new_aspect_column = column + '_' + str(i)
            aspect = [int(element[i]) for element in l.loc[:, column]]
            l[new_aspect_column] = aspect
            l[new_aspect_column] = l[new_aspect_column].astype(int)
        
    l.drop(aspect_columns, axis=1, inplace=True)
        
    return l


def train_forest(X, y):
    """
    Train a random forest regression model on input data X and labels y, then
    return the model for further processing.
    
    Arguments:
        X(DataFrame): input data
        y(DataFrame): labels
        
    Returns:
        forest_model(RandomForestRegressor): trained random forest models after cross-fold validation
    """
    # makes sure the models and grid search object give the same results every time
    random = 0
    num_iter = 10

    #instantiate models here
    forest = RandomForestRegressor(random_state=random)

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
    forest_model = forest_search.fit(X, y)
    print()
    
    return forest_model