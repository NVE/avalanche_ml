import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

#######################
#######################
#    Season Plots
#######################
#######################

def plot_season_weather(all_seasons, first_season, second_season, third_season):
    """
    Plot columns for temperature, precipitation, windspeed and freezing level for all seasons.
    
    Arguments:
        all_seasons(DataFrame): a dataframe containing all of the input data
        first_season(DataFrame): a dataframe containing the input data for the first season
        second_season(DataFrame): a dataframe containing the input data for the second season
        third_season(DataFrame): a dataframe containing the input data for the third season
    
    Returns:
        None
    """
    plot_list = ['temp_min_0', 'temp_max_0', 'precip_0', 'wind_speed_0', 'temp_lev_0', 'temp_freeze_lev_0']

    # we need to do this because the third season has some really high anomolous values for precip
    drop_condition_all = all_seasons['precip_0'].values != 0
    drop_condition_first = np.logical_and(first_season['precip_0'].values != 0, first_season['precip_0'].values < 5000)
    drop_condition_second = np.logical_and(second_season['precip_0'].values != 0, second_season['precip_0'].values < 5000)
    drop_condition_third = np.logical_and(third_season['precip_0'].values != 0, third_season['precip_0'].values < 5000)

    row, col = 0, 0

    plot_kwargs = {
        "alpha": 0.5,
        "rwidth":0.95,
        "bins":50,
    }

    fig, ax = plt.subplots(2, 3, figsize=(16, 12))

    # we are plotting the seasons in reverse order because the third season towers over the first and second
    for i, column in enumerate(plot_list):
        if column == 'precip_0':
            ax[row, col].hist(third_season[column].values[drop_condition_third], label='Third', **plot_kwargs)
            ax[row, col].hist(second_season[column].values[drop_condition_second], label='Second', **plot_kwargs)
            ax[row, col].hist(first_season[column].values[drop_condition_first], label='First', **plot_kwargs)

        # we can plot these values side by side because they are discrete, not continuous
        elif column == 'wind_speed_0' or column == 'temp_lev_0':
            plot_dfs = [third_season[column], second_season[column], first_season[column]]
            hist_labels = ['Third', 'Second', 'First']
            ax[row, col].hist(plot_dfs, bins=10, alpha=0.7, label=hist_labels)

        else:
            ax[row, col].hist(third_season[column], label='Third', **plot_kwargs)
            ax[row, col].hist(second_season[column], label='Second', **plot_kwargs)
            ax[row, col].hist(first_season[column], label='First', **plot_kwargs)

        ax[row, col].set_xlabel(' ')
        ax[row, col].set_title(column)
        ax[row, col].legend(bbox_to_anchor=(1.0, 1.0))

        # update row and column for to move to next plot
        if(col < 2):
            col += 1
        else: 
            col = 0

        if((i+1) % 3 == 0 and i > 0):
            row += 1

    ax[0, 0].set_ylabel('Count')
    ax[1, 0].set_ylabel('Count')
    plt.show()
    
    
def plot_season_danger_dsize(labels, means, column_list, total_rows, total_cols, size, start, end, how='values', rotate=False):
    """
    Plot the distribution of values in the labels for a different column.
    
    Arguments:
       labels(list): list of dataframes containing the labels for each season
       means(DataFrame): pandas dataframe object containing the means for each column
       column_list(list): the list of columns you would like to plot from the different dataframes
       total_rows(int): number of subplots along y axis
       total_cols(int): number of subplots along x axis
       size(tuple): two integers, how big the Matplotlib plot should be
       start(int): column number to start plotting on
       end(int): column number to end plotting on
       how(str): either 'values' or 'means'
       rotate(bool): whether or not to rotate the x axis labels, mainly use for when axis labels are long strings
       
    Note:
        start - end should equal total_rows x total_cols
       
    Returns:
       None
    """
    first_labels, second_labels, third_labels, fourth_labels = labels
    row = 0
    col = 0

    fig, ax = plt.subplots(total_rows, total_cols, figsize=size)

    for i, column in enumerate(column_list[start:end]):
        
        if how == 'values':
            list_of_dfs = [first_labels[column].values, second_labels[column].values,
                           third_labels[column].values, fourth_labels[column].values,]
            hist_labels = ['first', 'second', 'third', 'fourth',]
        elif how == 'means':
            list_of_dfs = [means.iloc[j][[column]] for j in range(5)]
            del list_of_dfs[3] # currently the fourth season does not have enough data and is skewing the graphs
            hist_labels = ['first', 'second', 'third', 'all']

        ax[row, col].hist(list_of_dfs, alpha=0.6, label=hist_labels)

        ax[row, col].set_title(column)
        ax[row, col].legend()
        
        # we need to draw the plot before using this method to rotate xtick labels
        if rotate == True:
            plt.draw()
            ax[row, col].set_xticklabels(ax[row, col].get_xticklabels(), rotation=30, ha='right')
        
        # update row and column for to move to next plot
        if(col < 2):
            col += 1
        else: 
            col = 0

        if((i+1) % 3 == 0 and i > 0):
            row += 1
    
    ax[2, 2].set_axis_off()
    plt.show()
    
    
def plot_season_levels(labels, real, classes, level, how='values'):
    """
    Plot the distribution of values for the level min and max for a given avalanche problem,
    after subsetting for the times when it lev_fill is not 0.
    
    Arguments:
       labels(list): list of dataframes containing the labels for each season
       real(list): lev_min and lev_max columns
       classes(list): lev_fill corresponding to every lev_min/lev_max column pair in real
       level(int): choices are 1, 2, and 4
       
    Returns:
       None
    """

    if level not in [1, 2, 4]:
        print('Please choose a valid integer for level fill: 1, 2, or 4')
        return
    
    first_labels, second_labels, third_labels, fourth_labels = labels
    row = 0
    col = 0

    fig, ax = plt.subplots(4, 2, figsize=(14, 20))

    for i, column in enumerate(classes):
        to_plot = []
        for df in labels:
            subset_cols = [column] + real[i]
            df_temp = df.loc[:, subset_cols]

            # replace lev_fill == 0 with NaNs for easy dropping
            df_temp[column].replace(0, np.nan, inplace=True)
            df_temp[column].dropna(inplace=True)

            # remove values where lev_fill == 3
            df_temp = df_temp[df_temp[column] != 3]
            
            # select for a specific level
            df_temp = df_temp.iloc[np.where(df_temp[column] == level)[0]]
            
            if how == 'means':
                df_means = df_temp[df_temp.columns[1:].values].mean()
                if df_means.isnull().all():
                    df_means.replace(np.nan, 0, inplace=True)

                to_plot.append(df_means)
            
            else:
                to_plot.append(df_temp[df_temp.columns[1:].values].values.flatten())
        
        hist_labels = ['First', 'Second', 'Third', 'Fourth']
        ax[row, col].hist(to_plot, alpha=0.6, label=hist_labels)

        title = column[6:-5]
        ax[row, col].set_title(title)
        ax[row, col].legend()

        # update row and column for to move to next plot
        if(col < 1):
            col += 1
        else: 
            col = 0

        if((i+1) % 2 == 0 and i > 0):
            row += 1
    
    ax[3, 1].set_axis_off()
    plt.show()
    

def plot_season_problems(labels, col_list):
    """
    Plot the distribution of values for a given avalanche problem,
    after subsetting for the times when it the columns *_prob are not 0
    for each season.
    
    Arguments:
       labels(list): list of dataframes containing the labels for each season
       col_list(list): list of columns pertaining to the avalanche problems
    
    Returns:
       None
    """
    first_labels, second_labels, third_labels, fourth_labels = labels
    row = 0
    col = 0

    fig, ax = plt.subplots(4, 3, figsize=(16, 18))

    for i, column in enumerate(col_list):
        to_plot = []
        for df in labels:
            
            # subset for column while keeping column name (DataFrame, not Series)
            # we will need the column name below
            df_temp = df.loc[:, col_list]
            
            # if the column is an avalanche problem, we want to select for times when it is actually given
            # i.e., not 0
            if column.endswith('_prob'):
                # replace lev_fill == 0 with NaNs for easy dropping
                df_temp[column].replace(0, np.nan, inplace=True)                
                df_temp[column].dropna(inplace=True)
                
            to_plot.append(df_temp[column].values)
        
        hist_labels = ['First', 'Second', 'Third', 'Fourth']
        ax[row, col].hist(to_plot, alpha=0.6, label=hist_labels)

        title = column[6:]
        ax[row, col].set_title(title)
        ax[row, col].legend()

        # update row and column for to move to next plot
        if(col < 2):
            col += 1
        else: 
            col = 0

        if((i+1) % 3 == 0 and i > 0):
            row += 1
                
    ax[3, 2].set_axis_off()
    plt.show()
    

#######################
#######################
#   Regional Plots
#######################
#######################

def plot_region_levels(labels, idx_list, real, classes, level, how='values'):
    """
    Plot the distribution of values for the level min and max for a given avalanche problem,
    after subsetting for the times when it lev_fill is not 0.
    
    Arguments:
       labels(DataFrame): a pandas dataframe object containing all of the labels
       idx_list(list): a list of regions id numbers that you want to analyze
       real(list): lev_min and lev_max columns
       classes(list): lev_fill corresponding to every lev_min/lev_max column pair in real
       level(int): choices are 1, 2, and 4
       
    Returns:
       None
    """

    if level not in [1, 2, 4]:
        print('Please choose a valid integer for level fill: 1, 2, or 4')
        return
    
    regions_df = [labels.loc[idx] for idx in idx_list]
    row = 0
    col = 0

    fig, ax = plt.subplots(4, 2, figsize=(14, 20))

    for i, column in enumerate(classes):
        to_plot = []
        for df in regions_df:
            subset_cols = [column] + real[i]
            df_temp = df.loc[:, subset_cols]

            # replace lev_fill == 0 with NaNs for easy dropping
            df_temp[column].replace(0, np.nan, inplace=True)
            df_temp[column].dropna(inplace=True)

            # remove values where lev_fill == 3
            df_temp = df_temp[df_temp[column] != 3]
            
            # select for a specific level
            df_temp = df_temp.iloc[np.where(df_temp[column] == level)[0]]
            
            if how == 'means':
                df_means = df_temp[df_temp.columns[1:].values].mean()
                if df_means.isnull().all():
                    df_means.replace(np.nan, 0, inplace=True)

                to_plot.append(df_means)
            
            else:
                to_plot.append(df_temp[df_temp.columns[1:].values].values.flatten())
        
        ax[row, col].hist(to_plot, alpha=0.6)

        title = column[6:-5]
        ax[row, col].set_title(title)

        # update row and column for to move to next plot
        if(col < 1):
            col += 1
        else: 
            col = 0

        if((i+1) % 2 == 0 and i > 0):
            row += 1
    
    ax[3, 1].set_axis_off()
    plt.show()
    
    
def plot_region_problems(labels, idx_list, col_list):
    """
    Plot the distribution of values for a given avalanche problem,
    after subsetting for the times when it the columns *_prob are not 0
    for each region.
    
    Arguments:
       labels(list): list of dataframes containing the labels for each region
       idx_list(list): a list of regions id numbers that you want to analyze
       col_list(list): list of columns pertaining to the avalanche problems
    
    Returns:
       None
    """
    regions_df = [labels.loc[idx] for idx in idx_list]
    row = 0
    col = 0

    fig, ax = plt.subplots(4, 3, figsize=(16, 18))

    for i, column in enumerate(col_list):
        to_plot = []
        for df in regions_df:

            # subset for column while keeping column name (DataFrame, not Series)
            # we will need the column name below
            df_temp = df.loc[:, col_list]

            # if the column is an avalanche problem, we want to select for times when it is actually given
            # i.e., not 0
            if column.endswith('_prob'):
                # replace lev_fill == 0 with NaNs for easy dropping
                df_temp[column].replace(0, np.nan, inplace=True)                
                df_temp[column].dropna(inplace=True)

            to_plot.append(df_temp[column].values)

        ax[row, col].hist(to_plot, alpha=0.6)

        title = column[6:]
        ax[row, col].set_title(title)

        # update row and column for to move to next plot
        if(col < 2):
            col += 1
        else: 
            col = 0

        if((i+1) % 3 == 0 and i > 0):
            row += 1

    ax[3, 2].set_axis_off()
    plt.show()

#######################
#######################
#  Data Split Plots
#######################
#######################

def plot_spatial_correlation(region_id, regions, idx_list, predict_regions, map_regions):
    """
    Plots the spatial correlation across all input columns between region_id
    and every other region in our dataset.
    
    Arguments:
       region_id(int): the region that you want to focus the correlations calculations on
       regions(DataFrame):
       idx_list(list): a list of regions id numbers that you want to analyze
       predict_regions(GeoDataFrame): contains the region ids, name of the region, and the Polygon for only the A regions
       map_regions(GeoDataFrame): contains the region ids, name of the region, and the Polygon for all regions
    
    Returns:
       None
    """
    # get region id, compute correlations
    region_idx = np.where(idx_list == region_id)[0][0]
    region = regions[region_idx]
    regions_corr_with = [region.corrwith(df).dropna() for df in regions]
    
    # create geo dataframe with mean correlations
    mean_corr_list = [np.mean(corr) for corr in regions_corr_with]
    col_name = 'corr_' + str(region_id)
    df_corr = pd.DataFrame({col_name: mean_corr_list}, index=idx_list)
    geo_corr = gpd.GeoDataFrame(pd.concat([predict_regions, df_corr], ignore_index=False, axis=1))
    
    # plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    map_regions.plot(facecolor='grey', edgecolor='k', alpha=0.5, linewidth=2.0, ax=ax)
    geo_corr.plot(column=col_name, legend=True, cmap='Reds', ax=ax)

    ax.set_xlabel('Longitude (° E/W)')
    ax.set_ylabel('Latitude (° N/S)')
    ax.set_title('Regional Correlation, ID {}'.format(region_id))

    plt.show()
    

def plot_split_weather(train, val, test):
    """
    Plot columns for temperature, precipitation, windspeed and freezing level for each of the 
    training, validation, and test sets.
    
    Arguments:
        train(DataFrame): a dataframe containing the input data for the training set
        val(DataFrame): a dataframe containing the input data for the validation set
        test(DataFrame): a dataframe containing the input data for the test set
    
    Returns:
        None
    """
    plot_list = ['temp_min_0', 'temp_max_0', 'precip_0', 'wind_speed_0', 'temp_lev_0', 'temp_freeze_lev_0']

    # we need to do this because the third season has some really high anomolous values for precip
    drop_condition_train = np.logical_and(train['precip_0'].values != 0, train['precip_0'].values < 5000)
    drop_condition_val = np.logical_and(val['precip_0'].values != 0, val['precip_0'].values < 5000)
    drop_condition_test = np.logical_and(test['precip_0'].values != 0, test['precip_0'].values < 5000)

    row, col = 0, 0

    plot_kwargs = {
        "alpha": 0.5,
        "rwidth":0.95,
        "bins":50,
        "density":True,
    }

    fig, ax = plt.subplots(2, 3, figsize=(16, 12))

    # we are plotting the seasons in reverse order because the third season towers over the first and second
    for i, column in enumerate(plot_list):
        if column == 'precip_0':
            ax[row, col].hist(train[column].values[drop_condition_train], label='Train', **plot_kwargs)
            ax[row, col].hist(val[column].values[drop_condition_val], label='Val', **plot_kwargs)
            ax[row, col].hist(test[column].values[drop_condition_test], label='Test', **plot_kwargs)

        # we can plot these values side by side because they are discrete, not continuous
        elif column == 'wind_speed_0' or column == 'temp_lev_0':
            plot_dfs = [train[column], val[column], test[column]]
            hist_labels = ['Train', 'Val', 'Test']
            ax[row, col].hist(plot_dfs, bins=10, alpha=0.7, label=hist_labels, density=True)

        else:
            ax[row, col].hist(train[column], label='Train', **plot_kwargs)
            ax[row, col].hist(val[column], label='Val', **plot_kwargs)
            ax[row, col].hist(test[column], label='Test', **plot_kwargs)

        ax[row, col].set_xlabel(' ')
        ax[row, col].set_title(column)
        ax[row, col].legend(bbox_to_anchor=(1.0, 1.0))

        # update row and column for to move to next plot
        if(col < 2):
            col += 1
        else: 
            col = 0

        if((i+1) % 3 == 0 and i > 0):
            row += 1

    ax[0, 0].set_ylabel('Count')
    ax[1, 0].set_ylabel('Count')
    plt.show()
    

def plot_split_danger_dsize(labels, column_list, total_rows, total_cols, size, start, end, how='values', rotate=False):
    """
    Plot the distribution of values in the labels for a different column for each data split.
    
    Arguments:
       labels(list): a list of dataframes containing the labels for the train, val, and test sets
       column_list(list): the list of columns you would like to plot from the different dataframes
       total_rows(int): number of subplots along y axis
       total_cols(int): number of subplots along x axis
       size(tuple): two integers, how big the Matplotlib plot should be
       start(int): column number to start plotting on
       end(int): column number to end plotting on
       how(str): either 'values' or 'means'
       rotate(bool): whether or not to rotate the x axis labels, mainly use for when axis labels are long strings
       
    Note:
        start - end should equal total_rows x total_cols
       
    Returns:
       None
    """
    train, val, test = labels
    row = 0
    col = 0

    fig, ax = plt.subplots(total_rows, total_cols, figsize=size)

    for i, column in enumerate(column_list[start:end]):
        
        list_of_dfs = [train[column].values, val[column].values, test[column].values]
        hist_labels = ['Train', 'Val', 'Test']
        ax[row, col].hist(list_of_dfs, alpha=0.6, label=hist_labels, density=True)

        ax[row, col].set_title(column)
        ax[row, col].legend()
        
        # we need to draw the plot before using this method to rotate xtick labels
        if rotate == True:
            plt.draw()
            ax[row, col].set_xticklabels(ax[row, col].get_xticklabels(), rotation=30, ha='right')
        
        # update row and column for to move to next plot
        if(col < 2):
            col += 1
        else: 
            col = 0

        if((i+1) % 3 == 0 and i > 0):
            row += 1
    
    ax[2, 2].set_axis_off()
    plt.show()
    
    
def plot_split_levels(labels, real, classes, level, how='values'):
    """
    Plot the distribution of values for the level min and max for a given avalanche problem,
    after subsetting for the times when it lev_fill is not 0.
    
    Arguments:
       labels(list): a list of dataframes containing the labels for the train, val, and test sets
       real(list): lev_min and lev_max columns
       classes(list): lev_fill corresponding to every lev_min/lev_max column pair in real
       level(int): choices are 1, 2, and 4
       
    Returns:
       None
    """

    if level not in [1, 2, 4]:
        print('Please choose a valid integer for level fill: 1, 2, or 4')
        return
    
    train, val, test = labels
    row = 0
    col = 0

    fig, ax = plt.subplots(4, 2, figsize=(14, 20))

    for i, column in enumerate(classes):
        to_plot = []
        for df in labels:
            subset_cols = [column] + real[i]
            df_temp = df.loc[:, subset_cols]

            # replace lev_fill == 0 with NaNs for easy dropping
            df_temp[column].replace(0, np.nan, inplace=True)
            df_temp[column].dropna(inplace=True)

            # remove values where lev_fill == 3
            df_temp = df_temp[df_temp[column] != 3]

            # select for a specific level
            df_temp = df_temp.iloc[np.where(df_temp[column] == level)[0]]
            if how == 'means':
                df_means = df_temp[df_temp.columns[1:].values].mean()
                if df_means.isnull().all():
                    df_means.replace(np.nan, 0, inplace=True)

                to_plot.append(df_means)

            else:
                to_plot.append(df_temp[df_temp.columns[1:].values].values.flatten())

        hist_labels = ['Train', 'Val', 'Test']
        ax[row, col].hist(to_plot, alpha=0.6, density=True, label=hist_labels)

        title = column[6:-5]
        ax[row, col].set_title(title)
        ax[row, col].legend()

        # update row and column for to move to next plot
        if(col < 1):
            col += 1
        else: 
            col = 0

        if((i+1) % 2 == 0 and i > 0):
            row += 1
    
    ax[3, 1].set_axis_off()
    plt.show()
    

def plot_split_problems(labels, col_list):
    """
    Plot the distribution of values for a given avlanche problem,
    after subsetting for the times when it the columns *_prob are not 0
    for each of the training, validation, and testing splits.
    
    Arguments:
       labels(list): list of dataframes containing the labels for each season
       col_list(list): list of columns pertaining to the avalanche problems

    Returns:
       None
    """
    train, val, test = labels
    row = 0
    col = 0

    fig, ax = plt.subplots(4, 3, figsize=(16, 18))

    for i, column in enumerate(col_list):
        to_plot = []
        for df in labels:
            
            # subset for column while keeping column name (DataFrame, not Series)
            # we will need the column name below
            df_temp = df.loc[:, col_list]
            
            # if the column is an avalanche problem, we want to select for times when it is actually given
            # i.e., not 0
            if column.endswith('_prob'):
                # replace lev_fill == 0 with NaNs for easy dropping
                df_temp[column].replace(0, np.nan, inplace=True)                
                df_temp[column].dropna(inplace=True)
                
            to_plot.append(df_temp[column].values)
        
        hist_labels = ['Train', 'Val', 'Test']
        ax[row, col].hist(to_plot, alpha=0.6, density=True, label=hist_labels)

        title = column[6:]
        ax[row, col].set_title(title)
        ax[row, col].legend()

        # update row and column for to move to next plot
        if(col < 2):
            col += 1
        else: 
            col = 0

        if((i+1) % 3 == 0 and i > 0):
            row += 1
                
    ax[3, 2].set_axis_off()
    plt.show()