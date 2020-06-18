# -*- coding: utf-8 -*-
"""Structures data in ML-friendly ways."""

import sys
import os
import copy
import datetime as dt
import time
import re
import numpy as np
import pandas
import requests
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "./varsomdata")
import setenvironment as se
from varsomdata import getforecastapi as gf
from varsomdata import getvarsompickles as gvp
from varsomdata import getmisc as gm


__author__ = 'arwi'

WIND_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

WIND_SPEEDS = {
    'Stille/svak vind': 0.,
    'Bris': 5.5,
    'Frisk bris': 9.,
    'Liten kuling': 12.,
    'Stiv kuling': 15.5,
    'Sterk kuling': 18.5,
    'Liten storm': 23.,
    'Storm': 30.
}

PROBLEMS = {
    3: 'new_loose',
    5: 'wet_loose',
    7: 'new_slab',
    10: 'drift_slab',
    30: 'pwl_slab',
    37: 'pwl_slab',
    45: 'wet_slab',
    50: 'glide'
}

CAUSES = {
    10: 'new_wl',
    11: 'rime',
    13: 'facet',
    14: 'ice',
    15: 'drift',
    16: 'gnd_facet',
    18: 'ice_a_facet ',
    19: 'ice_b_facet',
    20: 'gnd_water',
    22: 'water',
    24: 'loose'
}

TRIGGERS = {
    10: 0,
    21: 1,
    22: 2
}

REGOBS_CLASSES = {
    13: {
        "attr": "DangerSignTID",
        "categories": {
            2: 'avalanches',
            3: 'noise',
            4: 'cracks',
            5: 'snowfall',
            6: 'rime',
            7: 'temp_rise',
            8: 'water',
            9: 'wind_drift',
        }
    },
    25: {
        "attr": "PropagationTName",
        "categories": {
            "ECTPV": "ectpv",
            "ECTP": "ectp",
            "ECTN": "ectn",
            "ECTX": "ectx",
        }
    }
}

COMPETENCE = [0, 110, 115, 120, 130, 150]


class ForecastDataset:

    def __init__(self, seasons=('2017-18', '2018-19', '2019-20'), regobs_types=(13, 25)):
        """
        Object contains aggregated data used to generate labeled datasets.
        :param seasons: Tuple/list of string representations of avalanche seasons to fetch.
        :param regobs_types: Tuple/list of numerical IDs for RegObs observation types to fetch.
        """
        self.regobs_types = regobs_types
        self.tree = {}
        self.flat = []

        aw = []
        regobs = {}
        for season in seasons:
            aw += gvp.get_all_forecasts(year=season)
            regions = gm.get_forecast_regions(year=season, get_b_regions=True)
            regobs = {**regobs, **_get_regobs_obs(regions, season, regobs_types)}

        for forecast in aw:

            row = {
                # Metadata
                'region_id': forecast.region_id,
                'region_name': forecast.region_name,
                'region_type': forecast.region_type_name,
                'date': forecast.date_valid,

                'danger_level': forecast.danger_level,
                'emergency_warning': float(forecast.emergency_warning == 'Ikke gitt')
            }

            # Weather data
            weather = {
                'precip_most_exposed': forecast.mountain_weather.precip_most_exposed,
                'precip': forecast.mountain_weather.precip_region,
                'wind_speed': WIND_SPEEDS.get(forecast.mountain_weather.wind_speed, 0),
                'wind_change_speed': WIND_SPEEDS.get(forecast.mountain_weather.change_wind_speed, 0),
                'temp_min': forecast.mountain_weather.temperature_min,
                'temp_max': forecast.mountain_weather.temperature_max,
                'temp_lev': forecast.mountain_weather.temperature_elevation,
                'temp_freeze_lev': forecast.mountain_weather.freezing_level,
            }

            # We use multiple loops to get associated values near each other in e.g. .csv-files.
            for wind_dir in WIND_DIRECTIONS:
                weather[f"wind_dir_{wind_dir}"] = float(forecast.mountain_weather.wind_direction == wind_dir)
            for wind_dir in WIND_DIRECTIONS:
                weather[f"wind_chg_dir_{wind_dir}"] = float(forecast.mountain_weather.change_wind_direction == wind_dir)
            hours = [0, 6, 12, 18]
            for h in hours:
                weather[f"wind_chg_start_{h}"] = float(forecast.mountain_weather.change_hour_of_day_start == h)
            for h in hours:
                weather[f"temp_fl_start_{h}"] = float(forecast.mountain_weather.change_hour_of_day_start == h)
            row['weather'] = weather

            # Problem data
            prb = {}
            problem_types = [PROBLEMS.get(p.avalanche_problem_type_id, None) for p in forecast.avalanche_problems]
            problems = {}
            prb['problem_amount'] = len(forecast.avalanche_problems)
            for problem in PROBLEMS.values():
                if problem in problem_types:
                    index = problem_types.index(problem)
                    problems[problem] = forecast.avalanche_problems[index]
                    prb[f"problem_{problem}"] = -(problems[problem].avalanche_problem_id - 4)
                else:
                    problems[problem] = gf.AvalancheWarningProblem()
                    prb[f"problem_{problem}"] = 0
            for problem in PROBLEMS.values():
                p_data = problems[problem]
                for cause in CAUSES.values():
                    forecast_cause = CAUSES.get(p_data.aval_cause_id, None)
                    prb[f"problem_{problem}_cause_{cause}"] = float(forecast_cause == cause)
                prb[f"problem_{problem}_dsize"] = p_data.destructive_size_ext_id
                prb[f"problem_{problem}_prob"] = p_data.aval_probability_id
                prb[f"problem_{problem}_trig"] = TRIGGERS.get(p_data.aval_trigger_simple_id, 0)
                prb[f"problem_{problem}_dist"] = p_data.aval_distribution_id
                prb[f"problem_{problem}_lev_max"] = p_data.exposed_height_1
                prb[f"problem_{problem}_lev_min"] = p_data.exposed_height_2
                for n in range(0, 8):
                    aspect_attr_name = f"problem_{problem}_aspect_{WIND_DIRECTIONS[n]}"
                    prb[aspect_attr_name] = float(p_data.valid_expositions[n])

                # Check for consistency
                if prb[f"problem_{problem}_lev_min"] > prb[f"problem_{problem}_lev_max"]:
                    continue

            row['problems'] = prb

            # RegObs data
            row['regobs'] = regobs.get((forecast.region_id, forecast.date_valid), None)

            # Check for consistency
            if weather['temp_min'] > weather['temp_max']:
                continue

            self.tree[(forecast.region_id, forecast.date_valid)] = row
            self.flat.append(row)

    def label(self, days, b_regions=False, stars=0):
        """Creates a LabeledData containing relevant labels and features formatted either in a flat structure or as
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
        :param b_regions:       Include B-regions or not.
        :param stars:           How many stars must RegObs users have to be included?

        :return:                LabeledData
        """
        def regobs_transform(prev_n, id):
            value = 0
            try:
                obses = [x for x in prev_n["regobs"][id] if x["competence"] >= COMPETENCE[stars]]
                value = max(map(lambda x: x[cat_name], obses))
            except (TypeError, IndexError, ZeroDivisionError, KeyError):
                pass
            return value

        table = []
        labels = []
        days_w = {0: 1, 1: 1, 2: 1}.get(days, days - 1)
        days_v = {0: 1, 1: 2, 2: 2}.get(days, days)
        days_r = days + 1

        for entry in self.flat:
            prev = []
            if not b_regions and entry['region_type'] == 'B':
                continue
            try:
                for n in range(0, days_r):
                    prev.append(self.tree[(entry['region_id'], entry['date'] - dt.timedelta(days=n))])
            except KeyError:
                continue

            labels.append({'danger_level': entry['danger_level']})

            row = {}
            # It would obviously be better code-wise to flip the loops, but we need this insertion order.
            for key in prev[0]['weather'].keys():
                for n in range(0, days_w):
                    row[(key, n)] = prev[n]['weather'][key]
            for n in range(1, days_v):
                row[("danger_level", n)] = prev[n]['danger_level'] if n > 0 else 0
            for n in range(1, days_v):
                row[("emergency_warning", n)] = prev[n]['emergency_warning'] if n > 0 else 0
            for key in prev[0]['problems'].keys():
                for n in range(1, days_v):
                    row[(key, n)] = prev[n]['problems'][key] if n > 0 else 0
            for id in self.regobs_types:
                reg_type = REGOBS_CLASSES[id]
                for cat_name in reg_type["categories"].values():
                    for n in range(2, days_r):
                        attr_name = f"regobs_{_camel_to_snake(reg_type['attr'])}_{cat_name}"
                        row[(attr_name, n)] = regobs_transform(prev[n], id)
            table.append(row)

        df_label = pandas.DataFrame(labels)
        # We must force Pandas to create a MultiIndex
        table_dict = {}
        for idx, row in enumerate(table):
            table_dict[idx] = row
        df = pandas.DataFrame(table_dict).transpose()
        df = df.fillna(0)
        df = df.astype(np.float64)
        return LabeledData(df, df_label, days, self.regobs_types)


class LabeledData:
    is_normalized = False
    scaler = MinMaxScaler()

    def __init__(self, data, label, days, regobs_types):
        """Holds labels and features.

        :param data:            A DataFrame containing the features of the dataset.
        :param label:           DataFrame of labels.
        :param days:            How far back in time values should data be included.
                                If 0, only weather data for the forecast day is evaluated.
                                If 1, day 0 is used for weather, 1 for Varsom.
                                If 2, day 0 is used for weather, 1 for Varsom.
                                If 3, days 0-1 is used for weather, 1-2 for Varsom, 2-3 for RegObs.
                                If 5, days 0-3 is used for weather, 1-4 for Varsom, 2-5 for RegObs.
                                The reason for this is to make sure that each kind of data contain
                                the same number of data points, if we want to use some time series
                                frameworks that are picky about such things.
        :param regobs_types:    A tuple/list of observations to fetch from RegObs
        """
        self.data = data
        self.label = label
        self.days = days
        self.regobs_types = regobs_types
        self.scaler.fit(self.data)

    def normalize(self):
        """Normalize the data feature-wise using MinMax.

        :return: Normalized copy of LabeledData
        """
        if not self.is_normalized:
            ld = self.copy()
            data = ld.scaler.transform(self.data.values)
            ld.data = pandas.DataFrame(data=data, index=self.data.index, columns=self.data.columns)
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
            ld.data = pandas.DataFrame(data=data, index=self.data.index, columns=self.data.columns)
            ld.is_normalized = False
            return ld
        else:
            return self.copy()

    def to_timeseries(self):
        """Formats the data in a way that is parseable for e.g. `tslearn`. That is, a numpy array with
        shape `(rows, timeseries, modalities)`.

        :return: numpy.ndarray
        """
        shape = self.data.shape
        first = len(self.data.columns.get_level_values(0).unique())
        second = self.days - 1 if self.days >= 3 else 1
        return self.data.values.reshape(shape[0], first, second)

    def to_csv(self):
        """ Writes a csv-file in `varsomdata/localstorage` named according to the properties of the dataset.
        A `label.csv` is also always written.
        """
        # Write training data
        regobs = ""
        if len(self.regobs_types) and self.days > 2:
            regobs = f"_regobs_{'_'.join([str(type) for type in self.regobs_types])}"
        pathname_data = f"{se.local_storage}data_days_{self.days}{regobs}.csv"
        pathname_label = f"{se.local_storage}label_days_{self.days}{regobs}.csv"
        ld = self.denormalize()
        ld.data.to_csv(pathname_data, sep=';')
        ld.label.to_csv(pathname_label, sep=';')

    def copy(self):
        ld = LabeledData(self.data.copy(), self.label.copy(), self.days, copy.copy(self.regobs_types))
        ld.scaler = self.scaler
        return ld


def _get_regobs_obs(regions, year, requested_types, max_file_age=23):
    observations = {}

    if len(requested_types) == 0:
        return observations

    type_str = "_".join([str(type) for type in requested_types])
    file_name = f'{se.local_storage}regobs_{year}_{type_str}_lk1.pickle'
    file_date_limit = dt.datetime.now() - dt.timedelta(hours=max_file_age)
    current_season = gm.get_season_from_date(dt.date.today() - dt.timedelta(30))
    number_of_records = 50

    try:
        # Don't fetch new data if old is cached. If older season file doesn't exists we get out via an exception.
        if dt.datetime.fromtimestamp(os.path.getmtime(file_name)) > file_date_limit or year != current_season:
            with open(file_name, 'rb') as handle:
                return pickle.load(handle)
    except FileNotFoundError:
        pass

    from_date, to_date = gm.get_dates_from_season(year=year)

    if not set(REGOBS_CLASSES.keys()).issuperset(set(requested_types)):
        raise RegObsRegTypeError()

    url = "https://api.nve.no/hydrology/regobs/webapi_v3.2.0/Search/Avalanche"
    query = {
        "LangKey": 1,
        "FromDate": from_date.isoformat(),
        "ToDate": to_date.isoformat(),
        "SelectedRegistrationTypes": [],
        "SelectedRegions": regions,
        "NumberOfRecords": number_of_records,
        "Offset": 0
    }
    for requested_type in requested_types:
        query["SelectedRegistrationTypes"].append({
            "Id": requested_type,
            "SubTypes": []
        })

    while True:
        try:
            raw_obses = requests.post(url=url, json=query).json()
        except requests.exceptions.ConnectionError:
            time.sleep(1)

            continue

        for raw_obs in raw_obses["Results"]:
            for reg in raw_obs["Registrations"]:
                id = reg["RegistrationTid"]
                if id not in requested_types:
                    continue

                obs = {
                    "competence": raw_obs["CompetenceLevelTid"]
                }
                value = reg["FullObject"][REGOBS_CLASSES[id]["attr"]]
                for cat_id, cat_name in REGOBS_CLASSES[id]["categories"].items():
                    obs[cat_name] = 1 if cat_id == value else 0

                date = dt.datetime.fromisoformat(raw_obs["DtObsTime"]).date()
                key = (raw_obs["ForecastRegionTid"], date)
                if key not in observations:
                    observations[key] = {}
                if id not in observations[key]:
                    observations[key][id] = []
                observations[key][id].append(obs)

        query["Offset"] += number_of_records
        if raw_obses["ResultsInPage"] < number_of_records:
            break

    # We want the most competent observations first
    for date_region in observations.values():
        for reg_type in date_region.values():
            reg_type.sort(key=lambda x: x['competence'], reverse=True)

    with open(file_name, 'wb') as handle:
        pickle.dump(observations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return observations


_camel_re_1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re_2 = re.compile('([a-z0-9])([A-Z])')


def _camel_to_snake(name):
    name = _camel_re_1.sub(r'\1_\2', name)
    return _camel_re_2.sub(r'\1_\2', name).lower()


class Error(Exception):
    pass


class RegObsRegTypeError(Error):
    pass


if __name__ == '__main__':
    print("Fetching data (this will take a very long time the first run, then it is cached)")
    forecast_dataset = ForecastDataset(regobs_types=(13, 25))
    print("Labeling data")
    labeled_data = forecast_dataset.label(days=3)
    print("Writing .csv") # The .csv is always written using denormalized data.
    labeled_data.to_csv()
    print("Normalizing data")
    labeled_data = labeled_data.normalize()
    print("Transforming label")
    le = LabelEncoder()
    labels = labeled_data.label.values.ravel()
    le.fit(labels)
    labels = le.transform(labels)
    print("Running classifier")
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    f1 = []
    for split_idx, (train_index, test_index) in enumerate(kf.split(labeled_data.data)):
        X_train = labeled_data.data.iloc[train_index]
        X_test = labeled_data.data.iloc[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        clf = SVC()
        clf.fit(X_train, y_train)
        predictions = le.inverse_transform(clf.predict(X_test))
        danger = le.inverse_transform(y_test)
        f1_part = f1_score(danger, predictions, labels=le.classes_, average='weighted')
        f1.append(f1_part)
        print(f"F1 split {split_idx}: {f1_part}")
    print(f"F1 mean: {sum(f1) / len(f1)}")
