# -*- coding: utf-8 -*-
"""Structures data in ML-friendly ways."""

import re
import copy
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold

from avaml import Error, varsomdata, setenvironment as se, _NONE, CSV_VERSION, REGIONS, merge
from avaml.download import _get_varsom_obs, _get_weather_obs, _get_regobs_obs, REG_ENG

from varsomdata import getforecastapi as gf
from varsomdata import getmisc as gm


__author__ = 'arwi'

LABEL_PROBLEM_PRIMARY = {
    "ext_attr": [
        "avalanche_problem_type_id",
        "avalanche_problem_type_name",
        "avalanche_type_id",
        "avalanche_type_name",
        "avalanche_ext_id",
        "avalanche_ext_name"
    ],
    "values": {
        _NONE: [0, "", 0, "", 0, ""],
        "new-loose": [3, "Nysnø (løssnøskred)", 20, "Løssnøskred", 10, "Tørre løssnøskred"],
        "wet-loose": [5, "Våt snø (løssnøskred)", 20, "Løssnøskred", 15, "Våte løssnøskred"],
        "new-slab": [7, "Nysnø (flakskred)", 10, "Flakskred", 20, "Tørre flakskred"],
        "drift-slab": [10, "Fokksnø (flakskred)", 10, "Flakskred", 20, "Tørre flakskred"],
        "pwl-slab": [30, "Vedvarende svakt lag (flakskred)", 10, "Flakskred", 20, "Tørre flakskred"],
        "wet-slab": [45, "Våt snø (flakskred)", 10, "Flakskred", 25, "Våte flakskred"],
        "glide": [50, "Glideskred", 10, "Flakskred", 25, "Våte flakskred"]
    }
}

LABEL_PROBLEM = {
    "cause": {
        "ext_attr": ["aval_cause_id", "aval_cause_name"],
        "values": {
            "0": [0, ""],
            "new-snow": [10, "Nedføyket svakt lag med nysnø"],
            "hoar": [11, "Nedsnødd eller nedføyket overflaterim"],
            "facet": [13, "Nedsnødd eller nedføyket kantkornet snø"],
            "crust": [14, "Dårlig binding mellom glatt skare og overliggende snø"],
            "snowdrift": [15, "Dårlig binding mellom lag i fokksnøen"],
            "ground-facet": [16, "Kantkornet snø ved bakken"],
            "crust-above-facet": [18, "Kantkornet snø over skarelag"],
            "crust-below-facet": [19, "Kantkornet snø under skarelag"],
            "ground-water": [20, "Vann ved bakken/smelting fra bakken"],
            "water-layers": [22, "Opphopning av vann i/over lag i snødekket"],
            "loose": [24, "Ubunden snø"]
        }
    },
    "dsize": {
        "ext_attr": ["destructive_size_ext_id", "destructive_size_ext_name"],
        "values": {
            '0': [0, "Ikke gitt"],
            '1': [1, "1 - Små"],
            '2': [2, "2 - Middels"],
            '3': [3, "3 - Store"],
            '4': [4, "4 - Svært store"],
            '5': [5, "5 - Ekstremt store"]
        }
    },
    "prob": {
        "ext_attr": ["aval_probability_id", "aval_probability_name"],
        "values": {
            '0': [0, "Ikke gitt"],
            '2': [2, "Lite sannsynlig"],
            '3': [3, "Mulig"],
            '5': [5, "Sannsynlig"],
        }
    },
    "trig": {
        "ext_attr": ["aval_trigger_simple_id", "aval_trigger_simple_name"],
        "values": {
            '0': [0, "Ikke gitt"],
            '10': [10, "Stor tilleggsbelastning"],
            '21': [21, "Liten tilleggsbelastning"],
            '22': [22, "Naturlig utløst"]
        }
    },
    "dist": {
        "ext_attr": ["aval_distribution_id", "aval_distribution_name"],
        "values": {
            '0': [0, "Ikke gitt"],
            '1': [1, "Få bratte heng"],
            '2': [2, "Noen bratte heng"],
            '3': [3, "Mange bratte heng"],
            '4': [4, "De fleste bratte heng"]
        }
    },
    "lev_fill": {
        "ext_attr": ["exposed_height_fill"],
        "values": {
            '0': [0],
            '1': [1],
            '2': [2],
            '3': [3],
            '4': [4],
        }
    }
}

LABEL_PROBLEM_MULTI = {
    "aspect": {
        "ext_attr": "valid_expositions",
    }
}

LABEL_PROBLEM_REAL = {
    "lev_max": {
        "ext_attr": "exposed_height_1",
    },
    "lev_min": {
        "ext_attr": "exposed_height_2",
    }
}

LABEL_GLOBAL = {
    "danger_level": {
        "ext_attr": ["danger_level", "danger_level_name"],
        "values": {
            '1': [1, "1 liten"],
            '2': [2, "2 Moderat"],
            '3': [3, "3 Betydelig"],
            '4': [4, "4 Stor"],
            '5': [5, "5 Meget stor"]
        }
    },
    "emergency_warning": {
        "ext_attr": ["emergency_warning"],
        "values": {
            "Ikke gitt": ["Ikke gitt"],
            "Naturlig utløste skred": ["Naturlig utløste skred"],
        }
    }
}

COMPETENCE = [0, 110, 115, 120, 130, 150]


class ForecastDataset:

    def __init__(self, regobs_types, seasons=('2017-18', '2018-19', '2019-20')):
        """
        Object contains aggregated data used to generate labeled datasets.
        :param regobs_types: Tuple/list of string names for RegObs observation types to fetch.
        :param seasons: Tuple/list of string representations of avalanche seasons to fetch.
        """
        self.regobs_types = regobs_types
        self.weather = {}
        self.regobs = {}
        self.varsom = {}
        self.labels = {}

        for season in seasons:
            regions = gm.get_forecast_regions(year=season, get_b_regions=True)
            varsom, labels = _get_varsom_obs(year=season)
            self.varsom = merge(self.varsom, varsom)
            self.labels = merge(self.labels, labels)
            regobs = _get_regobs_obs(regions, season, regobs_types)
            self.regobs = merge(self.regobs, regobs)
            weather = _get_weather_obs(season)
            self.weather = merge(self.weather, weather)

    def label(self, days, with_varsom=True):
        """Creates a LabeledData containing relevant label and features formatted either in a flat structure or as
        a time series.

        :param days:            How far back in time values should data be included.
                                If 0, only weather data for the forecast day is evaluated.
                                If 1, day 0 is used for weather, 1 for Varsom.
                                If 2, day 0 is used for weather, 1 for Varsom.
                                If 3, days 0-1 is used for weather, 1-2 for Varsom, 2-3 for RegObs.
                                If 5, days 0-3 is used for weather, 1-4 for Varsom, 2-5 for RegObs.
                                The reason for this is to make sure that each kind of data contain
                                the same number of data points, if we want to use some time series
                                frameworks that are picky about such things.

        :param with_varsom:      Whether to include previous avalanche bulletins into the indata.

        :return:                LabeledData
        """
        table = {}
        row_weight = {}
        df = None
        df_weight = None
        df_label = pd.DataFrame(self.labels, dtype="U")
        days_w = {0: 1, 1: 1, 2: 1}.get(days, days - 1)
        days_v = {0: 1, 1: 2, 2: 2}.get(days, days)
        days_r = days + 1
        varsom_index = pd.DataFrame(self.varsom).index
        weather_index = pd.DataFrame(self.weather).index

        for monotonic_idx, entry_idx in enumerate(df_label.index):
            date, region_id = dt.date.fromisoformat(entry_idx[0]), entry_idx[1]

            def prev_key(day_dist):
                return (date - dt.timedelta(days=day_dist)).isoformat(), region_id

            # Just check that we can use this entry.
            try:
                if with_varsom:
                    for n in range(1, days_v):
                        if prev_key(n) not in varsom_index:
                            raise KeyError
                for n in range(0, days_w):
                    if prev_key(n) not in weather_index:
                        raise KeyError
                # We don't check for RegObs as it is more of the good to have type of data
            except KeyError:
                continue

            row = {}
            for region in REGIONS:
                row[(f"region_id_{region}", 0)] = float(region == region_id)

            if with_varsom:
                for column in self.varsom.keys():
                    for n in range(1, days_v):
                        row[(column, n)] = self.varsom[column][prev_key(n)]
            for column in self.weather.keys():
                for n in range(0, days_w):
                    try:
                        row[(column, n)] = self.weather[column][prev_key(n)]
                    except KeyError:
                        row[(column, n)] = 0
            for column in self.regobs.keys():
                for n in range(2, days_r):
                    try:
                        row[(column, n)] = self.regobs[column][prev_key(n)]
                    except KeyError:
                        row[(column, n)] = 0
            try:
                weight_sum = self.regobs['accuracy'][prev_key(0)]
                if weight_sum < 0:
                    row_weight[entry_idx] = 1 / 2
                elif weight_sum == 0:
                    row_weight[entry_idx] = 1
                elif weight_sum > 0:
                    row_weight[entry_idx] = 2
            except KeyError:
                row_weight[entry_idx] = 1

            # Some restructuring to make DataFrame parse the dict correctly
            for key in row.keys():
                if key not in table:
                    table[key] = {}
                table[key][entry_idx] = row[key]
            # Build DataFrame iteratively to preserve system memory (floats in dicts are apparently expensive).
            if (monotonic_idx > 0 and monotonic_idx % 1000 == 0) or monotonic_idx == len(df_label.index) - 1:
                df_new = pd.DataFrame(table, dtype=np.float32).fillna(0)
                df_weight_new = pd.Series(row_weight)
                df = df_new if df is None else pd.concat([df, df_new])
                df_weight = df_weight_new if df is None else pd.concat([df_weight, df_weight_new])
                table = {}
                row_weight = {}

        df_label = df_label.loc[df.index]

        df_label.sort_index(axis=0, inplace=True)
        df_label.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)
        df_weight.sort_index(axis=0, inplace=True)

        return LabeledData(df, df_label, df_weight, days, self.regobs_types, with_varsom)


class LabeledData:
    is_normalized = False
    scaler = MinMaxScaler()

    def __init__(self, data, label, row_weight, days, regobs_types, with_varsom):
        """Holds labels and features.

        :param data:            A DataFrame containing the features of the dataset.
        :param label:           DataFrame of labels.
        :param row_weight:      Series containing row weights
        :param days:            How far back in time values should data be included.
                                If 0, only weather data for the forecast day is evaluated.
                                If 1, day 0 is used for weather, 1 for Varsom.
                                If 2, day 0 is used for weather, 1 for Varsom, 2 for RegObs.
                                If 3, days 0-1 is used for weather, 1-2 for Varsom, 2-3 for RegObs.
                                If 5, days 0-3 is used for weather, 1-4 for Varsom, 2-5 for RegObs.
                                The reason for this is to make sure that each kind of data contain
                                the same number of data points, if we want to use some time series
                                frameworks that are picky about such things.
        :param regobs_types:    A tuple/list of strings of types of observations to fetch from RegObs.,
                                e.g., `("Faretegn")`.
        :param with_varsom:      Whether to include previous avalanche bulletins into the indata.
        """
        self.data = data
        self.row_weight = row_weight
        self.label = label
        self.label = self.label.replace(_NONE, 0)
        self.label = self.label.replace(np.nan, 0)
        try: self.label['CLASS', _NONE] = self.label['CLASS', _NONE].replace(0, _NONE).values
        except KeyError: pass
        try: self.label['MULTI'] = self.label['MULTI'].replace(0, "0").values
        except KeyError: pass
        try: self.label['REAL'] = self.label['REAL'].astype(np.float)
        except KeyError: pass
        self.pred = label.copy()
        for col in self.pred.columns:
            self.pred[col].values[:] = 0
        try: self.pred['CLASS', _NONE] = _NONE
        except KeyError: pass
        try: self.pred['MULTI'] = "0"
        except KeyError: pass
        self.days = days
        self.with_varsom = with_varsom
        self.regobs_types = regobs_types
        self.scaler.fit(self.data.values)

    def normalize(self):
        """Normalize the data feature-wise using MinMax.

        :return: Normalized copy of LabeledData
        """
        if not self.is_normalized:
            ld = self.copy()
            data = ld.scaler.transform(self.data.values)
            ld.data = pd.DataFrame(data=data, index=self.data.index, columns=self.data.columns)
            ld.is_normalized = True
            return ld
        else:
            return self.copy()

    def denormalize(self):
        """Denormalize the data feature-wise using MinMax.

        :return: Denormalized copy of LabeledData
        """
        if self.is_normalized:
            ld = self.copy()
            data = ld.scaler.inverse_transform(self.data.values)
            ld.data = pd.DataFrame(data=data, index=self.data.index, columns=self.data.columns)
            ld.is_normalized = False
            return ld
        else:
            return self.copy()

    def kfold(self, k=5, shuffle=True, stratify=None):
        """Returns an iterable of LabeledData-tuples. The first element is the training dataset
        and the second is for testing.

        :param k: Int: Number of folds.
        :param shuffle: Bool: Whether rows should be shuffled before folding. Defaults to True.
        :return: Iterable<(LabeledData, LabeledData)>
        """
        if stratify is None:
            kf = KFold(n_splits=k, shuffle=shuffle)
            split = kf.split(self.data)
        else:
            kf = StratifiedKFold(n_splits=k, shuffle=shuffle)
            split = kf.split(self.data, self.label[stratify])
        array = []
        for train_index, test_index in split:
            training_data = self.copy()
            training_data.data = training_data.data.iloc[train_index]
            training_data.label = training_data.label.iloc[train_index]
            training_data.pred = training_data.pred.iloc[train_index]
            training_data.row_weight = training_data.row_weight.iloc[train_index]
            testing_data = self.copy()
            testing_data.data = testing_data.data.iloc[test_index]
            testing_data.label = testing_data.label.iloc[test_index]
            testing_data.pred = testing_data.pred.iloc[test_index]
            testing_data.row_weight = testing_data.row_weight.iloc[test_index]
            array.append((training_data, testing_data))
        return array

    def f1(self):
        """Get F1, precision, recall and RMSE of all labels.

        :return: Series with scores of all possible labels and values.
        """
        dummies = self.to_dummies()
        old_settings = np.seterr(divide='raise', invalid='raise')

        class_types = []
        for typ in ["CLASS", "MULTI"]:
            if typ in dummies["label"].columns.get_level_values(0):
                class_types.append(typ)

        df = None
        if len(class_types):
            truth = dummies["label"][class_types]
            pred = dummies["pred"][class_types]
            true_pos = np.sum(truth * pred, axis=0)
            try:
                prec = true_pos / np.sum(pred, axis=0)
            except FloatingPointError:
                prec = pd.Series(index=pred.columns)
            try:
                recall = true_pos / np.sum(truth, axis=0)
            except FloatingPointError:
                recall = pd.Series(index=pred.columns)
            try:
                f1 = 2 * prec * recall / (prec + recall)
            except FloatingPointError:
                f1 = pd.Series(index=pred.columns)

            df = pd.DataFrame(index=truth.columns, columns=['f1', 'precision', 'recall', 'rmse'])
            df.iloc[:, :3] = np.array([f1, prec, recall]).transpose()
            df[['f1', 'precision', 'recall']] = df[['f1', 'precision', 'recall']].fillna(0)

        if "REAL" in dummies["label"].columns.get_level_values(0):
            truth = dummies["label"][["REAL"]]
            pred = dummies["pred"][["REAL"]]
            try:
                ntruth = (truth - truth.min(axis=0)) / (truth.max(axis=0) - truth.min(axis=0))
                npred = (pred - pred.min(axis=0)) / (pred.max(axis=0) - pred.min(axis=0))
                rmse = (np.sqrt(np.sum(np.square(npred - ntruth), axis=0)) / ntruth.shape[0])
            except Exception:
                rmse = np.nan
            rmse = pd.Series(rmse, index=truth.columns)
            df = rmse if df is None else pd.concat([df, rmse])
            np.seterr(**old_settings)

        return df

    def to_timeseries(self):
        """Formats the data in a way that is parseable for e.g. `tslearn`. That is, a numpy array with
        shape `(rows, timeseries, features)`.

        :return: (numpy.ndarray, list of feature names)
        """
        columns = self.data.columns.get_level_values(0).unique()
        number_of_features = len(columns)
        number_of_days = self.days - 1 if self.days >= 3 else 1
        shape = self.data.shape
        ts_array = np.zeros((shape[0], number_of_features * number_of_days), np.float64)
        # Multiply the region labels with the size of the time dimension.
        for idx in range(0, len(REGIONS)):
            for day in range(0, number_of_days + 1):
                ts_array[:, idx * number_of_days + day] = self.data.values[:, idx]
        ts_array[:, len(REGIONS) * number_of_days - 1:] = self.data.values[:, len(REGIONS) - 1:]
        ts_array = ts_array.reshape((shape[0], number_of_features, number_of_days))
        return ts_array.transpose((0, 2, 1)), columns

    def to_dummies(self):
        """Convert categorical variable into dummy/indicator variables.

        :return: pd.DataFrame
        """
        dummies = {}
        for name, df in [('label', self.label), ('pred', self.pred)]:
            dummies_types = {}
            dummies_class = {}
            for subprob in df.loc[:, ["CLASS"]].columns.get_level_values(1).unique():
                try:
                    sub_df = self.label["CLASS", subprob]
                    try: col = pd.get_dummies(sub_df, prefix_sep=':').columns
                    except ValueError: col = []

                    if name == 'label':
                        dum = pd.DataFrame(pd.get_dummies(sub_df, prefix_sep=':'), columns=col)
                        dummies_class[subprob] = dum.fillna(0)

                        columns = dummies_class[subprob].columns.values.astype("U")
                        idx = pd.MultiIndex.from_tuples(
                            [(a[0], a[2]) for a in np.char.partition(columns, sep=":")],
                            names=["attribute", "label"]
                        )
                        dummies_class[subprob].columns = idx
                    else:
                        dum = pd.DataFrame(pd.get_dummies(df["CLASS", subprob], prefix_sep=':'), columns=col)
                        dummies_class[subprob] = dum.fillna(0)

                        columns = dummies_class[subprob].columns.values.astype("U")
                        idx = pd.MultiIndex.from_tuples(
                            [(a[0], a[2]) for a in np.char.partition(columns, sep=":")],
                            names=["attribute", "label"]
                        )
                        dummies_class[subprob].columns = idx
                except KeyError:
                    pass
            dummies_types["CLASS"] = pd.concat(dummies_class.values(), keys=dummies_class.keys(), axis=1)

            dummies_multi = {}
            try:
                for subprob in df.loc[:, ['MULTI']].columns.get_level_values(1).unique():
                    try:
                        multi = df['MULTI'][subprob].replace(_NONE, "0").values.astype(np.int).astype("U")
                        if name == 'label':
                            multimax = np.max(np.char.str_len(multi), axis=0)
                        multi = np.char.zfill(multi, multimax)
                        multi = np.nan_to_num(np.array([[list(elem) for elem in row] for row in multi]))
                        multi = multi.reshape(multi.shape[0], multi.shape[1] * multi.shape[2]).astype(np.float)
                        columns = zip(df["MULTI"][subprob].columns, multimax)
                        columns = [[(c, str(n)) for n in range(max)] for c, max in columns]
                        columns = [item for sublist in columns for item in sublist]
                        columns = pd.MultiIndex.from_tuples(columns, names=["attribute", "label"])
                        dummies_multi[subprob] = pd.DataFrame(multi, index=df.index, columns=columns)
                    except KeyError:
                        pass
                dummies_types["MULTI"] = pd.concat(dummies_multi.values(), keys=dummies_multi.keys(), axis=1)
            except (KeyError, ValueError):
                pass

            dummies_real = {}
            try:
                for subprob in df.loc[:, ["REAL"]].columns.get_level_values(1).unique():
                    try:
                        columns = pd.MultiIndex.from_tuples(
                            [(a, "") for a in df["REAL"][subprob].columns],
                            names=["attribute", "label"]
                        )
                        dummies_real[subprob] = pd.DataFrame(
                            df['REAL'][subprob].values,
                            columns=columns,
                            index=df.index
                        )
                    except KeyError:
                        pass
                dummies_types["REAL"] = pd.concat(dummies_real.values(), keys=dummies_real.keys(), axis=1)
            except (KeyError, ValueError):
                pass

            dummies[name] = pd.concat(dummies_types.values(), keys=dummies_types.keys(), axis=1)
        return pd.concat(dummies.values(), keys=dummies.keys(), axis=1)

    def to_csv(self):
        """ Writes a csv-file in `varsomdata/localstorage` named according to the properties of the dataset.
        A `label.csv` is also always written.
        """
        # Write training data
        regobs = ""
        if len(self.regobs_types) and self.days >= 2:
            regobs = f"_regobs_{'_'.join([REG_ENG[obs_type] for obs_type in self.regobs_types])}"
        varsom = "" if self.with_varsom else "_novarsom"
        pathname_data = f"{se.local_storage}data_v{CSV_VERSION}_days_{self.days}{regobs}{varsom}.csv"
        pathname_label = f"{se.local_storage}label_v{CSV_VERSION}_days_{self.days}{regobs}{varsom}.csv"
        pathname_weight = f"{se.local_storage}weight_v{CSV_VERSION}_days_{self.days}{regobs}{varsom}.csv"
        ld = self.denormalize()
        ld.data.to_csv(pathname_data, sep=';')
        ld.label.to_csv(pathname_label, sep=';')
        ld.row_weight.to_csv(pathname_weight, sep=';', header=False)

    def to_aw(self):
        """Convert predictions to AvalancheWarnings.

        :return: AvalancheWarning[]
        """
        aws = []
        for name, row in self.pred.iterrows():
            aw = gf.AvalancheWarning()
            aw.region_id = int(name[1])
            aw.valid_from = dt.datetime.combine(dt.date.fromisoformat(name[0]), dt.datetime.min.time())
            aw.valid_to = dt.datetime.combine(dt.date.fromisoformat(name[0]), dt.datetime.max.time())
            aw.mountain_weather = gf.MountainWeather()
            for int_attr, dict in LABEL_GLOBAL.items():
                for idx, ext_attr in enumerate(dict['ext_attr']):
                    try:
                        ext_val = dict['values'][row['CLASS', '', int_attr]][idx]
                        setattr(aw, ext_attr, ext_val)
                    except KeyError:
                        pass
            try:
                for p_idx in [1, 2, 3]:
                    p_prefix = f"problem_{p_idx}"
                    try:
                        p_name = row['CLASS', '', p_prefix]
                    except KeyError:
                        continue
                    if p_name == "":
                        break
                    problem = gf.AvalancheWarningProblem()
                    problem.avalanche_problem_id = -p_idx + 4
                    for idx, ext_attr in enumerate(LABEL_PROBLEM_PRIMARY['ext_attr']):
                        try:
                            ext_val = LABEL_PROBLEM_PRIMARY['values'][row['CLASS', '', p_prefix]][idx]
                            setattr(problem, ext_attr, ext_val)
                        except KeyError: pass
                    for int_attr, dict in LABEL_PROBLEM.items():
                        for idx, ext_attr in enumerate(dict['ext_attr']):
                            try:
                                ext_val = dict['values'][row['CLASS', p_name, int_attr]][idx]
                                setattr(problem, ext_attr, ext_val)
                            except KeyError: pass
                    for int_attr, dict in LABEL_PROBLEM_MULTI.items():
                        try:
                            ext_attr = dict['ext_attr']
                            ext_val = row['MULTI', p_name, int_attr]
                        except KeyError: pass
                        setattr(problem, ext_attr, ext_val)
                    for int_attr, dict in LABEL_PROBLEM_REAL.items():
                        try:
                            ext_attr = dict['ext_attr']
                            ext_val = row['REAL', p_name, int_attr]
                            setattr(problem, ext_attr, ext_val)
                        except KeyError: pass
                    aw.avalanche_problems.append(problem)
                aws.append(aw)
            except KeyError:
                pass
        return aws

    def copy(self):
        """Deep copy LabeledData.
        :return: copied LabeledData
        """
        ld = LabeledData(
            self.data.copy(deep=True),
            self.label.copy(deep=True),
            self.row_weight.copy(deep=True),
            self.days,
            copy.copy(self.regobs_types),
            self.with_varsom
        )
        ld.is_normalized = self.is_normalized
        ld.scaler = self.scaler
        ld.pred = self.pred.copy(deep=True)
        return ld

    @staticmethod
    def from_csv(days, regobs_types, with_varsom=True):
        """Read LabeledData from previously written .csv-file.

        :param days:            How far back in time values should data be included.
        :param regobs_types:    A tuple/list of strings of types of observations to fetch from RegObs.,
                                e.g., `("Faretegn")`.
        """
        regobs = ""
        if len(regobs_types) and days >= 2:
            regobs = f"_regobs_{'_'.join([REG_ENG[obs_type] for obs_type in regobs_types])}"
        varsom = "" if with_varsom else "_novarsom"
        pathname_data = f"{se.local_storage}data_v{CSV_VERSION}_days_{days}{regobs}{varsom}.csv"
        pathname_label = f"{se.local_storage}label_v{CSV_VERSION}_days_{days}{regobs}{varsom}.csv"
        pathname_weight = f"{se.local_storage}weight_v{CSV_VERSION}_days_{days}{regobs}{varsom}.csv"
        try:
            data = pd.read_csv(pathname_data, sep=";", header=[0, 1], index_col=[0, 1])
            label = pd.read_csv(pathname_label, sep=";", header=[0, 1, 2], index_col=[0, 1], low_memory=False, dtype="U")
            row_weight = pd.read_csv(pathname_weight, sep=";", header=None, index_col=[0, 1], low_memory=False, squeeze=True)
            columns = [(col[0], re.sub(r'Unnamed:.*', _NONE, col[1]), col[2]) for col in label.columns.tolist()]
            label.columns = pd.MultiIndex.from_tuples(columns)
        except FileNotFoundError:
            raise CsvMissingError()
        return LabeledData(data, label, row_weight, days, regobs_types, with_varsom)


class CsvMissingError(Error):
    pass