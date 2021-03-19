import re

import pandas as pd
import numpy as np

from avaml.aggregatedata import PROBLEMS
from avaml.aggregatedata.download import CAUSES, REG_ENG_V4, REGOBS_CLASSES, _camel_to_snake, REGOBS_SCALARS


def coeff(series):
    x = np.arange(series.shape[0])
    y = series.values
    return np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0][0]


def mode(series):
    return series.mode().iloc[0]


real_funcs = ['min', 'max', 'mean', 'median', 'std', coeff]
discrete_funcs = real_funcs + [mode]
binary_funcs = ['median', 'mean', coeff]
regobs_discrete_funcs = ['sum', coeff]
regobs_scalar_funcs = ['max', coeff]

real_columns = {
    'precip',
    'precip_most_exposed',
    'temp_freeze_lev',
    'temp_lev',
    'temp_max',
    'temp_min',
    'wind_change_speed',
    'wind_speed'
}
discrete_columns = {
    'danger_level',
    'problem_new-loose',
    'problem_wet-loose',
    'problem_new-slab',
    'problem_drift-slab',
    'problem_pwl-slab',
    'problem_wet-slab',
    'problem_glide',
    'problem_amount',
}
binary_columns = {
    'wind_dir_N',
    'wind_dir_NE',
    'wind_dir_E',
    'wind_dir_SE',
    'wind_dir_S',
    'wind_dir_SW',
    'wind_dir_W',
    'wind_dir_NW',
    'wind_chg_dir_N',
    'wind_chg_dir_NE',
    'wind_chg_dir_E',
    'wind_chg_dir_SE',
    'wind_chg_dir_S',
    'wind_chg_dir_SW',
    'wind_chg_dir_W',
    'wind_chg_dir_NW',
    'wind_chg_start_0',
    'wind_chg_start_6',
    'wind_chg_start_12',
    'wind_chg_start_18',
    'temp_fl_start_0',
    'temp_fl_start_6',
    'temp_fl_start_12',
    'temp_fl_start_18'
    'emergency_warning',
}
for prob in PROBLEMS.values():
    discrete_columns = discrete_columns.union({f"problem_{prob}_{attr}" for attr in ['dsize', 'prob', 'trig', 'dist']})
    binary_columns = binary_columns.union({f"problem_{prob}_cause_{cause}" for cause in CAUSES.values()})

regobs_discrete_columns = set()
for reg_type, reg_eng in REG_ENG_V4.items():
    for reg_class, subclasses in REGOBS_CLASSES[reg_type].items():
        reg_class = _camel_to_snake(reg_class)
        for subclass in subclasses.values():
            subclass = _camel_to_snake(subclass)
            for n in range(0, 5):
                col = f"regobs_{reg_eng}_{reg_class}_{subclass}_{n}"
                regobs_discrete_columns = regobs_discrete_columns.union({col})
regobs_scalar_columns = set()
for reg_type, reg_eng in REG_ENG_V4.items():
    for reg_scalar in REGOBS_SCALARS[reg_type].keys():
        reg_scalar = _camel_to_snake(reg_scalar)
        for n in range(0, 5):
            col = f"regobs_{reg_eng}_{reg_scalar}_{n}"
            regobs_scalar_columns = regobs_scalar_columns.union({col})


def to_time_parameters(labeled_data):
    labeled_data = labeled_data.drop_regions()
    data = labeled_data.data

    real_groups = data.loc[:, real_columns.intersection(data.columns.get_level_values(0))].T.groupby(level=0)
    discrete_groups = data.loc[:, discrete_columns.intersection(data.columns.get_level_values(0))].T.groupby(level=0)
    binary_groups = data.loc[:, binary_columns.intersection(data.columns.get_level_values(0))].T.groupby(level=0)
    regobs_discrete_groups = data.loc[
        :, regobs_discrete_columns.intersection(data.columns.get_level_values(0))
    ].T.groupby(by=lambda x: x[0][:-1] + str(x[1])).sum()
    regobs_discrete_groups = regobs_discrete_groups.groupby(by=lambda x: x[:-2])
    regobs_scalar_groups = data.loc[
        :, regobs_scalar_columns.intersection(data.columns.get_level_values(0))
    ].T.groupby(by=lambda x: x[0][:-1] + str(x[1])).max()
    regobs_scalar_groups = regobs_scalar_groups.groupby(by=lambda x: x[:-2])

    real_groups_params = real_groups.agg(real_funcs).T.unstack()
    discrete_groups_params = discrete_groups.agg(discrete_funcs).T.unstack()
    binary_groups_params = binary_groups.agg(binary_funcs).T.unstack()
    regobs_discrete_groups_params = regobs_discrete_groups.agg(regobs_discrete_funcs).T.unstack()
    regobs_scalar_groups_params = regobs_scalar_groups.agg(regobs_scalar_funcs).T.unstack()

    return pd.concat([
        real_groups_params,
        discrete_groups_params,
        binary_groups_params,
        regobs_discrete_groups_params,
        regobs_scalar_groups_params,
    ], axis=1).sort_index(axis=1)
