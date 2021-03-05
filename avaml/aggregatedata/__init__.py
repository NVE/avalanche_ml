# -*- coding: utf-8 -*-
"""Structures data in ML-friendly ways."""
import re
import copy
import datetime as dt
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from avaml import Error, setenvironment as se, _NONE, CSV_VERSION, REGIONS, merge
from avaml.aggregatedata.download import _get_varsom_obs, _get_weather_obs, _get_regobs_obs, REG_ENG, PROBLEMS
from avaml.aggregatedata.time_parameters import to_time_parameters
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

    def __init__(self, regobs_types, seasons=('2017-18', '2018-19', '2019-20'), max_file_age=23):
        """
        Object contains aggregated data used to generate labeled datasets.
        :param regobs_types: Tuple/list of string names for RegObs observation types to fetch.
        :param seasons: Tuple/list of string representations of avalanche seasons to fetch.
        """
        self.seasons = sorted(list(set(seasons)))
        self.date = None
        self.regobs_types = regobs_types
        self.weather = {}
        self.regobs = {}
        self.varsom = {}
        self.labels = {}
        self.use_label = True

        for season in seasons:
            varsom, labels = _get_varsom_obs(year=season, max_file_age=max_file_age)
            self.varsom = merge(self.varsom, varsom)
            self.labels = merge(self.labels, labels)
            regobs = _get_regobs_obs(season, regobs_types, max_file_age=max_file_age)
            self.regobs = merge(self.regobs, regobs)
            weather = _get_weather_obs(season, max_file_age=max_file_age)
            self.weather = merge(self.weather, weather)

    @staticmethod
    def date(regobs_types, date: dt.date, days, use_label=True):
        """
        Create a dataset containing just a given day's data.
        :param regobs_types: Tuple/list of string names for RegObs observation types to fetch.
        :param date: Date to fetch and create dataset for.
        :param days: How many days to fetch before date. This will be max for .label()'s days parameter.
        """
        self = ForecastDataset(regobs_types, [])
        self.date = date
        self.use_label = use_label

        self.regobs = _get_regobs_obs(None, regobs_types, date=date, days=days)
        self.varsom, labels = _get_varsom_obs(None, date=date, days=days-1 if days > 0 else 1)
        self.weather = _get_weather_obs(None, date=date, days=days-2 if days > 2 else 1)

        self.labels = {}
        for label_keys, label in labels.items():
            if label_keys not in self.labels:
                self.labels[label_keys] = {}
            for (label_date, label_region), label_data in label.items():
                if label_date == date.isoformat():
                    subkey = (label_date, label_region)
                    self.labels[label_keys][subkey] = label_data

        return self

    def label(self, days, with_varsom=True):
        """Creates a LabeledData containing relevant label and features formatted either in a flat structure or as
        a time series.

        :param days:            How far back in time values should data be included.
                                If 0, only weather data for the forecast day is evaluated.
                                If 1, day 0 is used for weather, 1 for Varsom.
                                If 2, day 0 is used for weather, 1 for Varsom, 2 for RegObs.
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

        if len(df_label.index) == 0 and self.use_label:
            raise NoBulletinWithinRangeError()

        if self.date and not self.use_label:
            season = gm.get_season_from_date(self.date)
            regions = gm.get_forecast_regions(year=season, get_b_regions=True)
            date_region = [(self.date.isoformat(), region) for region in regions]
        else:
            date_region = df_label.index

        for monotonic_idx, entry_idx in enumerate(date_region):
            date, region_id = dt.date.fromisoformat(entry_idx[0]), entry_idx[1]

            def prev_key(day_dist):
                return (date - dt.timedelta(days=day_dist)).isoformat(), region_id

            # Just check that we can use this entry.
            try:
                if with_varsom:
                    for n in range(1, days_v):
                        if prev_key(n) not in varsom_index:
                            raise KeyError()
                for n in range(0, days_w):
                    if prev_key(n) not in weather_index:
                        raise KeyError()
                add_row = True
                # We don't check for RegObs as it is more of the good to have type of data
            except KeyError:
                add_row = False

            if add_row:
                row = {}
                for region in REGIONS:
                    row[(f"region_id_{region}", "0")] = float(region == region_id)

                if with_varsom:
                    for column in self.varsom.keys():
                        for n in range(1, days_v):
                            # We try/except an extra time since single dates may run without a forecast.
                            row[(column, str(n))] = self.varsom[column][prev_key(n)]
                for column in self.weather.keys():
                    for n in range(0, days_w):
                        try:
                            row[(column, str(n))] = self.weather[column][prev_key(n)]
                        except KeyError:
                            row[(column, str(n))] = 0
                for column in self.regobs.keys():
                    for n in range(2, days_r):
                        try:
                            row[(column, str(n))] = self.regobs[column][prev_key(n)]
                        except KeyError:
                            row[(column, str(n))] = 0
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
            if (monotonic_idx > 0 and monotonic_idx % 1000 == 0) or monotonic_idx == len(date_region) - 1:
                df_new = pd.DataFrame(table, dtype=np.float32).fillna(0)
                df_weight_new = pd.Series(row_weight)
                df = df_new if df is None else pd.concat([df, df_new])
                df_weight = df_weight_new if df is None else pd.concat([df_weight, df_weight_new])
                table = {}
                row_weight = {}

        if df is None or len(df.index) == 0:
            raise NoDataFoundError()

        if self.use_label:
            df_label = df_label.loc[df.index]

            df_label.sort_index(axis=0, inplace=True)
            df_label.sort_index(axis=1, inplace=True)
            df.sort_index(axis=0, inplace=True)
            df_weight.sort_index(axis=0, inplace=True)
        else:
            df_label = None

        return LabeledData(df, df_label, df_weight, days, self.regobs_types, with_varsom, self.seasons)


class LabeledData:
    is_normalized = False
    scaler = StandardScaler()

    def __init__(self, data, label, row_weight, days, regobs_types, with_varsom, seasons=False):
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
        if label is not None:
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
        else:
            self.label = None
            self.pred = None
        self.days = days
        self.with_varsom = with_varsom
        self.regobs_types = regobs_types
        if self.data is not None:
            self.scaler.fit(self.data.values)
        self.single = not seasons
        self.seasons = sorted(list(set(seasons if seasons else [])))
        self.with_regions = True

    def normalize(self, by=None):
        """Normalize the data feature-wise using MinMax.

        :return: Normalized copy of LabeledData
        """
        by = by if by is not None else self
        if not self.is_normalized:
            ld = self.copy()
            data = by.scaler.transform(self.data.values)
            ld.data = pd.DataFrame(data=data, index=self.data.index, columns=self.data.columns)
            ld.is_normalized = by
            return ld
        elif self.is_normalized != by:
            return self.denormalize().normalize(by=by)
        else:
            return self.copy()

    def denormalize(self):
        """Denormalize the data feature-wise using MinMax.

        :return: Denormalized copy of LabeledData
        """
        if self.is_normalized:
            ld = self.copy()
            data = self.is_normalized.scaler.inverse_transform(self.data.values)
            ld.data = pd.DataFrame(data=data, index=self.data.index, columns=self.data.columns)
            ld.is_normalized = False
            return ld
        else:
            return self.copy()

    def drop_regions(self):
        """Remove regions from input data"""
        if self.with_regions:
            ld = self.copy()
            region_columns = list(filter(lambda x: re.match(r'^region_id', x[0]), ld.data.columns))
            ld.data.drop(region_columns, axis=1, inplace=True)
            ld.with_regions = False
            ld.scaler.fit(ld.data.values)
            return ld
        else:
            return self.copy()

    def stretch_temperatures(self):
        """Stretch out temperatures near zero"""
        ld = self.copy()
        if self.data is not None:
            temp_cols = [bool(re.match(r"^temp_(max|min)$", title)) for title in ld.data.columns.get_level_values(0)]

            ld.data.loc[:, temp_cols] = np.sign(ld.data.loc[:, temp_cols]) * np.sqrt(np.abs(ld.data.loc[:, temp_cols]))
            ld.scaler.fit(ld.data.values)
        return ld

    def problem_graph(self):
        label = pd.Series(self.label["CLASS", _NONE, "problem_1"], name="label")
        pred1 = pd.Series(self.pred["CLASS", _NONE, "problem_1"], name="problem_1")
        pred2 = pd.Series(self.pred["CLASS", _NONE, "problem_2"], name="problem_2")

        groups = pd.concat([label, pred1, pred2], axis=1).groupby(["label", "problem_1"], dropna=False)
        count = groups.count()["problem_2"].rename("count")
        p2 = groups["problem_2"].apply(lambda x: pd.Series.mode(x)[0]).replace(0, np.nan)
        return pd.concat([count, p2], axis=1)

    def statham(self):
        """Make a danger level in the same manner as Statham et al., 2018."""
        if self.pred is None:
            raise NotPredictedError

        label = self.label[("CLASS", _NONE, "danger_level")].apply(np.int)
        pred = self.pred[("CLASS", _NONE, "danger_level")].apply(np.int)
        ones = pd.Series(np.ones(pred.shape), index=pred.index)
        cols = ["label", "diff", "n"]
        df = pd.DataFrame(pd.concat([label, label - pred, ones], axis=1).values, columns=cols)
        bias = df.groupby(cols[:-1]).count().unstack().droplevel(0, axis=1)
        n = df.groupby(cols[0]).count()["n"]
        share = bias.divide(n, axis=0)
        return pd.concat([n, share], axis=1)

    def adam(self):
        if self.pred is None:
            raise NotPredictedError

        touch = pd.DataFrame({
            1: {(2, 10): "A", (3, 10): "A", (3, 21): "B", (5, 21): "B", (3, 22): "B", (5, 22): "B"},
            2: {(2, 10): "A", (3, 10): "B", (3, 21): "C", (5, 21): "D", (3, 22): "C", (5, 22): "D"},
            3: {(2, 10): "B", (3, 10): "C", (3, 21): "D", (5, 21): "E", (3, 22): "D", (5, 22): "E"},
            4: {(2, 10): "B", (3, 10): "C", (3, 21): "D", (5, 21): "E", (3, 22): "D", (5, 22): "E"}
        })
        danger = pd.DataFrame({
            1: {"A": 1, "B": 1, "C": 1, "D": 2, "E": 3},
            2: {"A": 1, "B": 2, "C": 2, "D": 3, "E": 4},
            3: {"A": 2, "B": 2, "C": 3, "D": 3, "E": 4},
            4: {"A": 2, "B": 3, "C": 4, "D": 4, "E": 5},
            5: {"A": 2, "B": 3, "C": 4, "D": 4, "E": 5}
        })

        def get_danger(series):
            p1 = series["CLASS", _NONE, "problem_1"]
            p2 = series["CLASS", _NONE, "problem_2"]
            p3 = series["CLASS", _NONE, "problem_2"]
            dl = ("CLASS", _NONE, "danger_level")
            ew = ("CLASS", _NONE, "emergency_warning")
            if p1 == _NONE:
                series[dl] = "1"
                series[ew] = "Ikke gitt"
            else:
                p1 = series["CLASS", p1][["prob", "trig", "dist", "dsize"]].apply(np.int)
                try:
                    dl1 = str(danger.loc[touch.loc[(p1["prob"], p1["trig"]), p1["dist"]], p1["dsize"]])
                except KeyError:
                    dl1 = 0

                if p2 != _NONE:
                    p2 = series["CLASS", p2][["prob", "trig", "dist", "dsize"]].apply(np.int)
                    try:
                        dl1 = str(danger.loc[touch.loc[(p1["prob"], p1["trig"]), p1["dist"]], p1["dsize"]])
                    except KeyError:
                        series[dl] = "2"

                series[ew] = "Ikke gitt"
                try:
                    if p1["trig"] == 22 and p1["dsize"] >= 3:
                        series[ew] = "Naturlig utløste skred"
                except KeyError:
                    pass

            return series

        ld = self.copy()
        ld.pred = ld.pred.apply(get_danger, axis=1)
        return ld

    def rangeify_elevations(self):
        """Convert all elevations to ranges"""
        MAX_ELEV = 2500

        def convert_label(df):
            problems = df.columns.get_level_values(1).unique().to_series().replace(_NONE, np.nan).dropna()
            for problem in problems:
                fill = df["CLASS", problem, "lev_fill"].apply(str)
                ones = fill == "1"
                twos = fill == "2"
                threes = fill == "3"

                df.loc[ones, ("REAL", problem, "lev_min")] = df.loc[ones, ("REAL", problem, "lev_max")]
                df.loc[ones, ("REAL", problem, "lev_max")] = MAX_ELEV
                df.loc[ones, ("CLASS", problem, "lev_fill")] = "4"

                df.loc[twos, ("REAL", problem, "lev_min")] = 0
                df.loc[twos, ("CLASS", problem, "lev_fill")] = "4"

                df.loc[threes, ("REAL", problem, "lev_min")] = 0
                df.loc[threes, ("REAL", problem, "lev_max")] = MAX_ELEV
                df.loc[threes, ("CLASS", problem, "lev_fill")] = "4"

        def convert_data(df):
            prefixes = set(map(lambda y: (y[0][:-7], y[1]), filter(lambda x: re.search(r"lev_fill", x[0]), df.columns)))
            for prefix in prefixes:
                ones = df[(f"{prefix[0]}_fill_1", prefix[1])].apply(np.bool)
                twos = df[(f"{prefix[0]}_fill_2", prefix[1])].apply(np.bool)
                threes = df[(f"{prefix[0]}_fill_3", prefix[1])].apply(np.bool)
                fours = df[(f"{prefix[0]}_fill_4", prefix[1])].apply(np.bool)

                df.loc[ones, (f"{prefix[0]}_min", prefix[1])] = df.loc[ones, (f"{prefix[0]}_max", prefix[1])]
                df.loc[ones, (f"{prefix[0]}_max", prefix[1])] = MAX_ELEV
                df.loc[ones == True, (f"{prefix[0]}_fill_4", prefix[1])] = 1
                df[(f"{prefix[0]}_fill_1", prefix[1])] = np.zeros(ones.shape)

                df.loc[twos, (f"{prefix[0]}_min", prefix[1])] = 0
                df.loc[twos == True, (f"{prefix[0]}_fill_4", prefix[1])] = 1
                df[(f"{prefix[0]}_fill_2", prefix[1])] = np.zeros(twos.shape)

                df.loc[threes, (f"{prefix[0]}_min", prefix[1])] = 0
                df.loc[threes, (f"{prefix[0]}_min", prefix[1])] = MAX_ELEV
                df.loc[threes == True, (f"{prefix[0]}_fill_4", prefix[1])] = 1
                df[(f"{prefix[0]}_fill_3", prefix[1])] = np.zeros(threes.shape)

        ld = self.copy().denormalize()
        if self.label is not None:
            convert_label(ld.label)
        if self.pred is not None:
            convert_label(ld.pred)
        if self.data is not None:
            convert_data(ld.data)

        if self.is_normalized:
            return ld.normalize()
        else:
            return ld

    def valid_pred(self):
        """Makes the bulletins internally coherent. E.g., removes problem 3 if problem 2 is blank."""
        if self.pred is None:
            raise NotPredictedError

        ld = self.copy()

        # Handle Problem 1-3
        prob_cols = []
        for n in range(1, 4):
            if f"problem_{n}" in list(ld.pred["CLASS", _NONE].columns):
                prob_cols.append(("CLASS", _NONE, f"problem_{n}"))
        prev_eq = np.zeros((ld.pred.shape[0], len(prob_cols)), dtype=bool)
        for n, col in enumerate(prob_cols):
            for mcol in prob_cols[0:n]:
                # If equal to problem_n-1/2, set to _NONE.
                prev_eq[:, n] = np.logical_or(
                    prev_eq[:, n],
                    np.equal(ld.pred[mcol], ld.pred[col])
                )
                # Set to None if problem_n-1/2 was _NONE.
                prev_eq[:, n] = np.logical_or(
                    prev_eq[:, n],
                    ld.pred[mcol] == _NONE
                )
            ld.pred.loc[prev_eq[:, n], col] = _NONE

        # Delete subproblem solutions that are irrelevant
        for subprob in PROBLEMS.values():
            rows = np.any(np.char.equal(ld.pred.loc[:, prob_cols].values.astype("U"), subprob), axis=1) == False
            columns = [name == subprob for name in ld.pred.columns.get_level_values(1)]
            ld.pred.loc[rows, columns] = _NONE

        # Set problem_amount to the right number
        ld.pred['CLASS', _NONE, 'problem_amount'] = np.sum(ld.pred.loc[:, prob_cols] != _NONE, axis=1).astype(str)

        # If lev_fill is "3" or "4", lev_min is always "0"
        for subprob in PROBLEMS.values():
            if "lev_fill" in ld.pred["CLASS", subprob].columns:
                fill = ld.pred.astype(str)["CLASS", subprob, "lev_fill"]
                if "lev_min" in ld.pred["REAL", subprob]:
                    ld.pred.loc[np.logical_or(fill == "1", fill == "2"), ("REAL", subprob, "lev_min")] = "0"
            if "lev_min" in ld.pred["REAL", subprob] and "lev_max" in ld.pred["REAL", subprob]:
                real = ld.pred["REAL", subprob].replace("", np.nan).astype(np.float)
                reversed_idx = real["lev_min"] > real["lev_max"]
                average = real.loc[reversed_idx, "lev_min"] + real.loc[reversed_idx, "lev_max"] / 2
                ld.pred.loc[reversed_idx, ("REAL", subprob, "lev_min")] = average
                ld.pred.loc[reversed_idx, ("REAL", subprob, "lev_max")] = average

        ld.pred.loc[:, ["CLASS", "MULTI"]] = ld.pred.loc[:, ["CLASS", "MULTI"]].astype(str)
        ld.pred["REAL"] = ld.pred["REAL"].replace("", np.nan).astype(np.float)
        return ld

    def split(self, rounds=3, seed="Njunis"):
        """Returns a split of the object into a training set, a test set and a validation set.

        Use as:
            for test, train, eval in ld.split():
                model.fit(test)
                model.predict(train)
                model.predict(eval)
        """
        rand = random.Random(seed)

        result = []
        for _ in range(0, rounds):
            troms = rand.sample([3006, 3007, 3009, 3010, 3011, 3012, 3013], k=7)
            nordland = rand.sample([3014, 3015, 3016, 3017], k=4)
            south = rand.sample([3022, 3023, 3024, 3027, 3028, 3029, 3031, 3032, 3034, 3035, 3037], k=11)

            train_regions = troms[2:] + nordland[2:] + south[2:]
            test_regions = [troms[0], nordland[0], south[0]]
            eval_regions = [troms[1], nordland[1], south[1]]

            split = []
            for regions in [train_regions, test_regions, eval_regions]:
                ld = self.copy()
                ld.data = ld.data.iloc[[region in regions for region in ld.data.index.get_level_values(1)]]
                ld.label = ld.label.iloc[[region in regions for region in ld.label.index.get_level_values(1)]]
                ld.pred = ld.pred.iloc[[region in regions for region in ld.pred.index.get_level_values(1)]]
                ld.row_weight = ld.row_weight.iloc[[region in regions for region in ld.row_weight.index.get_level_values(1)]]
                split.append(ld)
            result.append(tuple(split))

        return result


    def f1(self):
        """Get F1, precision, recall and RMSE of all labels.

        :return: Series with scores of all possible labels and values.
        """
        if self.label is None or self.pred is None:
            raise DatasetMissingLabel()

        dummies = self.to_dummies()
        old_settings = np.seterr(divide='ignore', invalid='ignore')

        df_idx = pd.MultiIndex.from_arrays([[], [], [], []])
        df = pd.DataFrame(index=df_idx, columns=["f1", "precision", "recall", "rmse"])

        try:
            prob_cols = [
                name.startswith("problem_") for name in self.label.columns.get_level_values(2)
            ]
        except KeyError:
            prob_cols = pd.DataFrame(index=self.label.index)
        for column, pred_series in dummies["pred"].items():
            if column[1]:
                true_idx = self.label.loc[
                    np.any(np.char.equal(self.label.loc[:, prob_cols].values.astype("U"), column[1]), axis=1)
                ].index
                pred_idx = self.pred.loc[
                    np.any(np.char.equal(self.pred.loc[:, prob_cols].values.astype("U"), column[1]), axis=1)
                ].index
                idx = list(set(true_idx.to_list()).intersection(set(pred_idx.to_list())))
            else:
                idx = list(set(self.label.index).intersection(set(self.pred.index)))

            if column[0] in ["CLASS", "MULTI"] and column in dummies["label"].columns:
                truth = dummies["label"][column][idx]
                pred = pred_series[idx]
                true_pos = np.sum(truth * pred)

                if not np.sum(truth) or (column[0] == "CLASS" and column[1] and column[3] == "0"):
                    continue

                prec = true_pos / np.sum(pred) if np.sum(pred) else 0
                recall = true_pos / np.sum(truth)
                f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0

                df.loc[column] = pd.Series([f1, prec, recall, np.nan], index=df.columns)
            elif column[0] in ["REAL"] and column in dummies["label"].columns:
                truth = dummies["label"][column][idx]
                pred = pred_series[idx]

                if not len(truth):
                    continue

                rmse = np.sqrt(np.sum(np.square(pred - truth))) / len(truth)

                df.loc[column] = pd.Series([np.nan, np.nan, np.nan, rmse], index=df.columns)

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

    def to_time_parameters(self, orig_days=-1):
        """Collapses the time series to fewer dimensions"""
        ld = self.copy()
        ld.data = pd.concat([
            ld.data.loc[:, ld.data.columns.get_level_values(1).values.astype(int) <= orig_days],
            to_time_parameters(ld)
        ], axis=1).sort_index()
        ld.scaler.fit(ld.data.values)
        return ld

    def to_dummies(self):
        """Convert categorical variable into dummy/indicator variables.

        :return: pd.DataFrame
        """
        if self.label is None:
            raise DatasetMissingLabel()

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
        return pd.concat(dummies.values(), keys=dummies.keys(), axis=1).replace("", np.nan).astype(np.float)

    def to_csv(self, tag=""):
        """ Writes a csv-file in `varsomdata/localstorage` named according to the properties of the dataset.
        A `label.csv` is also always written.
        """
        regobs = ""
        if len(self.regobs_types) and self.days >= 2:
            regobs = f"_regobs_{'--'.join([REG_ENG[obs_type] for obs_type in self.regobs_types])}"
        varsom = "" if self.with_varsom else "_novarsom"
        tag_ = "_" + tag if tag else ""
        if self.single:
            pathname_data = f"{se.local_storage}single_data_v{CSV_VERSION}{tag_}_days_{self.days}{regobs}{varsom}.csv"
            pathname_label = f"{se.local_storage}single_label_v{CSV_VERSION}{tag_}_days_{self.days}{regobs}{varsom}.csv"
            pathname_weight = f"{se.local_storage}single_weight_v{CSV_VERSION}{tag_}_days_{self.days}{regobs}{varsom}.csv"
            try:
                old_ld = LabeledData.from_csv(
                    self.days,
                    self.regobs_types,
                    with_varsom=self.with_varsom,
                    seasons=self.seasons,
                    tag=tag,
                )
                ld = self.denormalize()
                ld.data = pd.concat([old_ld.data, ld.data], axis=0)
                ld.row_weight = pd.concat([old_ld.row_weight, ld.row_weight], axis=0)
                unique = np.unique(ld.data.index)
                ld.data = ld.data.loc[unique]
                ld.row_weight = ld.row_weight.loc[unique]
                if old_ld.label is not None and ld.label is not None:
                    ld.label = pd.concat([old_ld.label, ld.label], axis=0)
                if ld.label is not None:
                    ld.label = ld.label.loc[unique]
            except CsvMissingError:
                ld = self.denormalize()
        else:
            seasons = "--".join(self.seasons)
            pathname_data = f"{se.local_storage}data_v{CSV_VERSION}{tag_}_days_{self.days}{regobs}{varsom}_{seasons}.csv"
            pathname_label = f"{se.local_storage}label_v{CSV_VERSION}{tag_}_days_{self.days}{regobs}{varsom}_{seasons}.csv"
            pathname_weight = f"{se.local_storage}weight_v{CSV_VERSION}{tag_}_days_{self.days}{regobs}{varsom}_{seasons}.csv"
            ld = self.denormalize()

        ld.data.to_csv(pathname_data, sep=';')
        ld.row_weight.to_csv(pathname_weight, sep=';', header=False)
        if ld.label is not None:
            ld.label.to_csv(pathname_label, sep=';')

    def to_aw(self):
        """Convert predictions to AvalancheWarnings.

        :return: AvalancheWarning[]
        """
        if self.label is None or self.pred is None:
            raise DatasetMissingLabel()

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
            self.data.copy(deep=True) if self.data is not None else None,
            self.label.copy(deep=True) if self.label is not None else None,
            self.row_weight.copy(deep=True),
            self.days,
            copy.copy(self.regobs_types),
            self.with_varsom,
            self.seasons
        )
        ld.is_normalized = self.is_normalized
        ld.scaler = self.scaler
        ld.pred = self.pred.copy(deep=True) if self.pred is not None else None
        return ld

    @staticmethod
    def from_csv(days, regobs_types, seasons=('2017-18', '2018-19', '2019-20'), with_varsom=True, tag=""):
        """Read LabeledData from previously written .csv-file.

        :param days:            How far back in time values should data be included.
        :param regobs_types:    A tuple/list of strings of types of observations to fetch from RegObs.,
                                e.g., `("Faretegn")`.
        """
        single = not seasons
        seasons = "--".join(sorted(list(set(seasons if seasons else []))))
        tag = "_" + tag if tag else ""
        regobs = ""
        if len(regobs_types) and days >= 2:
            regobs = f"_regobs_{'--'.join([REG_ENG[obs_type] for obs_type in regobs_types])}"
        varsom = "" if with_varsom else "_novarsom"
        if single:
            pathname_data = f"{se.local_storage}single_data_v{CSV_VERSION}{tag}_days_{days}{regobs}{varsom}.csv"
            pathname_label = f"{se.local_storage}single_label_v{CSV_VERSION}{tag}_days_{days}{regobs}{varsom}.csv"
            pathname_weight = f"{se.local_storage}single_weight_v{CSV_VERSION}{tag}_days_{days}{regobs}{varsom}.csv"
        else:
            pathname_data = f"{se.local_storage}data_v{CSV_VERSION}{tag}_days_{days}{regobs}{varsom}_{seasons}.csv"
            pathname_label = f"{se.local_storage}label_v{CSV_VERSION}{tag}_days_{days}{regobs}{varsom}_{seasons}.csv"
            pathname_weight = f"{se.local_storage}weight_v{CSV_VERSION}{tag}_days_{days}{regobs}{varsom}_{seasons}.csv"
        try:
            label = pd.read_csv(pathname_label, sep=";", header=[0, 1, 2], index_col=[0, 1], low_memory=False, dtype="U")
            columns = [(col[0], re.sub(r'Unnamed:.*', _NONE, col[1]), col[2]) for col in label.columns.tolist()]
            label.columns = pd.MultiIndex.from_tuples(columns)
        except FileNotFoundError:
            label = None
        try:
            data = pd.read_csv(pathname_data, sep=";", header=[0, 1], index_col=[0, 1])
            row_weight = pd.read_csv(pathname_weight, sep=";", header=None, index_col=[0, 1], low_memory=False, squeeze=True)
            data.columns = pd.MultiIndex.from_tuples(data.columns)
        except FileNotFoundError:
            raise CsvMissingError()
        return LabeledData(data, label, row_weight, days, regobs_types, with_varsom, seasons)


class CsvMissingError(Error):
    pass

class SingleDateCsvError(Error):
    pass

class NoBulletinWithinRangeError(Error):
    pass

class NoDataFoundError(Error):
    pass

class DatasetMissingLabel(Error):
    pass

class NotPredictedError(Error):
    pass
