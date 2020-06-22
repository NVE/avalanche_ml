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
    "Faretegn": {
        "DangerSignTID": {
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
    "Tester": {
        "PropagationTName": {
            "ECTPV": "ectpv",
            "ECTP": "ectp",
            "ECTN": "ectn",
            "ECTX": "ectx",
            "LBT": "lbt",
            "CTV": "ctv",
            "CTE": "cte",
            "CTM": "ctm",
            "CTH": "cth",
            "CTN": "ctn",
        },
        "ComprTestFractureTName": {
            "Q1": "q1",
            "Q2": "q2",
            "Q3": "q3",
        },
    },
    "Skredaktivitet": {
        "AvalancheExtTID": {
            10: "dry_loose",
            15: "wet_loose",
            20: "dry_slab",
            25: "wet_slab",
            27: "glide",
            30: "slush",
            40: "cornice",
        },
        "ExposedHeightComboTID": {
            1: "blackup",
            2: "blackbottom",
            3: "blackupbottom",
            4: "blackmiddle",
        },
    }
}

REGOBS_SCALARS = {
    "Faretegn": {},
    "Tester": {
        "FractureDepth": ("FractureDepth", lambda x: x),
        "TapsFracture": ("TapsFracture", lambda x: x),
        "StabilityEval": ("StabilityEvalTID", lambda x: x),
    },
    "Skredaktivitet": {
        "EstimatedNum": ("EstimatedNumTID", lambda x: x),
        "AvalTrigger": ("AvalTriggerSimpleTID", lambda x: {22: 5, 60: 4, 50: 3, 40: 2, 30: 1}.get(x, 0)),
        "DestructiveSize": ("DestructiveSizeTID", lambda x: x if 0 < x <= 5 else 0),
        "AvalPropagation": ("AvalPropagationTID", lambda x: x),
        "ExposedHeight1": ("ExposedHeight1", lambda x: x),
        "ExposedHeight2": ("ExposedHeight2", lambda x: x),
        "ValidExpositionN": ("ValidExposition", lambda x: float(x[0])),
        "ValidExpositionNE": ("ValidExposition", lambda x: float(x[1])),
        "ValidExpositionE": ("ValidExposition", lambda x: float(x[2])),
        "ValidExpositionSE": ("ValidExposition", lambda x: float(x[3])),
        "ValidExpositionS": ("ValidExposition", lambda x: float(x[4])),
        "ValidExpositionSW": ("ValidExposition", lambda x: float(x[5])),
        "ValidExpositionW": ("ValidExposition", lambda x: float(x[6])),
        "ValidExpositionNW": ("ValidExposition", lambda x: float(x[7])),
    },
}

COMPETENCE = [0, 110, 115, 120, 130, 150]

REGIONS = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018,
           3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036,
           3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046]


class ForecastDataset:

    def __init__(self, regobs_types, seasons=('2017-18', '2018-19', '2019-20')):
        """
        Object contains aggregated data used to generate labeled datasets.
        :param regobs_types: Tuple/list of string names for RegObs observation types to fetch.
        :param seasons: Tuple/list of string representations of avalanche seasons to fetch.
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
        :param b_regions:       Include B-regions or not.
        :param stars:           How many stars must RegObs users have to be included?

        :return:                LabeledData
        """
        table = []
        label = []
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

            label.append({'danger_level': entry['danger_level']})

            row = {}
            for region in REGIONS:
                row[(f"region_id_{region}", 0)] = float(region == entry["region_id"])

            # It would obviously be better code-wise to flip the loops, but we need this insertion order.
            for key in entry['weather'].keys():
                for n in range(0, days_w):
                    row[(key, n)] = prev[n]['weather'][key]
            for n in range(1, days_v):
                row[("danger_level", n)] = prev[n]['danger_level'] if n > 0 else 0
            for n in range(1, days_v):
                row[("emergency_warning", n)] = prev[n]['emergency_warning'] if n > 0 else 0
            for key in entry['problems'].keys():
                for n in range(1, days_v):
                    row[(key, n)] = prev[n]['problems'][key] if n > 0 else 0

            # Filter out low-competence obses
            def obses(obs_type):
                return [x for x in obs_type if x["competence"] >= COMPETENCE[stars]]
            i = 0
            # Use 5 most competent observations, and list both categories as well as scalars
            while i < 5:
                # One type of observation (test, danger signs etc.) at a time
                for regobs_type in self.regobs_types:
                    # Go through each requested class attribute from the specified observation type
                    for attr, cat in REGOBS_CLASSES[regobs_type].items():
                        # We handle categories using 1-hot, so we step through each category
                        for cat_name in cat.values():
                            for n in range(2, days_r):
                                value = 0
                                try:
                                    value = obses(prev[n]["regobs"][regobs_type])[i][cat_name]
                                except (TypeError, IndexError, ZeroDivisionError, KeyError):
                                    pass
                                attr_name = f"regobs_{_camel_to_snake(attr)}_{cat_name}_{i}"
                                row[(attr_name, n)] = value
                    # Go through all requested scalars
                    for attr, (regobs_attr, conv) in REGOBS_SCALARS[regobs_type].items():
                        for n in range(2, days_r):
                            value = 0
                            try:
                                value = conv(obses(prev[n]["regobs"][regobs_type])[i][regobs_attr])
                            except (TypeError, IndexError, ZeroDivisionError, KeyError):
                                pass
                            attr_name = f"regobs_{_camel_to_snake(attr)}_{i}"
                            row[(attr_name, n)] = value
                i += 1
            table.append(row)

        df_label = pandas.DataFrame(label)
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
                                If 2, day 0 is used for weather, 1 for Varsom, 2 for RegObs.
                                If 3, days 0-1 is used for weather, 1-2 for Varsom, 2-3 for RegObs.
                                If 5, days 0-3 is used for weather, 1-4 for Varsom, 2-5 for RegObs.
                                The reason for this is to make sure that each kind of data contain
                                the same number of data points, if we want to use some time series
                                frameworks that are picky about such things.
        :param regobs_types:    A tuple/list of strings of types of observations to fetch from RegObs.,
                                e.g., `("Faretegn")`.
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
        shape `(rows, timeseries, features)`.

        :return: (numpy.ndarray, list of feature names)
        """
        headings = self.data.columns.get_level_values(0).unique()
        number_of_features = len(headings)
        number_of_days = self.days - 1 if self.days >= 3 else 1
        shape = self.data.shape
        ts_array = np.zeros((shape[0], number_of_features * number_of_days), np.float64)
        # Multiply the region labels with the size of the time dimension.
        for idx in range(0, len(REGIONS)):
            for day in range(0, number_of_days + 1):
                ts_array[:, idx * number_of_days + day] = self.data.values[:, idx]
        ts_array[:, len(REGIONS) * number_of_days - 1:] = self.data.values[:, len(REGIONS) - 1:]
        ts_array = ts_array.reshape((shape[0], number_of_features, number_of_days))
        return ts_array.transpose((0, 2, 1)), headings

    def to_csv(self):
        """ Writes a csv-file in `varsomdata/localstorage` named according to the properties of the dataset.
        A `label.csv` is also always written.
        """
        # Write training data
        regobs = ""
        if len(self.regobs_types) and self.days > 2:
            regobs = f"_regobs_{'_'.join([_camel_to_snake(obs_type) for obs_type in self.regobs_types])}"
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

    file_name = f'{se.local_storage}regobs_{year}.pickle'
    file_date_limit = dt.datetime.now() - dt.timedelta(hours=max_file_age)
    current_season = gm.get_season_from_date(dt.date.today() - dt.timedelta(30))
    number_of_records = 50
    get_new = True

    try:
        # Don't fetch new data if old is cached. If older season file doesn't exists we get out via an exception.
        if dt.datetime.fromtimestamp(os.path.getmtime(file_name)) > file_date_limit or year != current_season:
            get_new = False
    except FileNotFoundError:
        pass

    from_date, to_date = gm.get_dates_from_season(year=year)

    req_set = set(requested_types)
    class_set = set(list(REGOBS_CLASSES.keys()))
    scalar_set = set(list(REGOBS_SCALARS.keys()))
    if not class_set.issuperset(req_set) or not scalar_set.issuperset(req_set):
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

    response = []
    if get_new:
        print(f"Fetching {query['Offset']}-{query['Offset'] + query['NumberOfRecords']}")
        while True:
            try:
                raw_obses = requests.post(url=url, json=query).json()
            except requests.exceptions.ConnectionError:
                time.sleep(1)
                continue
            response = response + raw_obses["Results"]

            query["Offset"] += number_of_records
            if raw_obses["ResultsInPage"] < number_of_records:
                with open(file_name, 'wb') as handle:
                    pickle.dump(response, handle, protocol=pickle.HIGHEST_PROTOCOL)
                break
            print(f"{query['Offset']}-{query['Offset'] + query['NumberOfRecords']}/{raw_obses['TotalMatches']}")
    else:
        with open(file_name, 'rb') as handle:
            response = pickle.load(handle)

    for raw_obs in response:
        for reg in raw_obs["Registrations"]:
            obs_type = reg["RegistrationName"]
            if obs_type not in requested_types:
                continue

            obs = {
                "competence": raw_obs["CompetenceLevelTid"]
            }
            try:
                for attr, categories in REGOBS_CLASSES[obs_type].items():
                    value = reg["FullObject"][attr]
                    for cat_id, cat_name in categories.items():
                        obs[cat_name] = 1 if cat_id == value else 0
            except KeyError:
                pass
            try:
                for regobs_attr, conv in REGOBS_SCALARS[obs_type].values():
                    obs[regobs_attr] = reg["FullObject"][regobs_attr]
            except KeyError:
                pass

            date = dt.datetime.fromisoformat(raw_obs["DtObsTime"]).date()
            key = (raw_obs["ForecastRegionTid"], date)
            if key not in observations:
                observations[key] = {}
            if obs_type not in observations[key]:
                observations[key][obs_type] = []
            observations[key][obs_type].append(obs)

    # We want the most competent observations first
    for date_region in observations.values():
        for reg_type in date_region.values():
            reg_type.sort(key=lambda x: x['competence'], reverse=True)

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
    forecast_dataset = ForecastDataset(regobs_types=("Faretegn", "Tester", "Skredaktivitet"))
    print("Labeling data")
    labeled_data = forecast_dataset.label(days=3)
    print("Writing .csv")  # The .csv is always written using denormalized data.
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
