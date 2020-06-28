# -*- coding: utf-8 -*-
"""Structures data in ML-friendly ways."""

import copy
import datetime as dt
import os
import pickle
import re
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas
import requests
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, "./varsomdata")
import setenvironment as se
from varsomdata import getforecastapi as gf
from varsomdata import getvarsompickles as gvp
from varsomdata import getmisc as gm


__author__ = 'arwi'

_pwl = re.compile("(DH|SH|FC)")

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

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
    3: 'new-loose',
    5: 'wet-loose',
    7: 'new-slab',
    10: 'drift-slab',
    30: 'pwl-slab',
    37: 'dpwl-slab',
    45: 'wet-slab',
    50: 'glide'
}

CAUSES = {
    10: 'new-snow',
    11: 'hoar',
    13: 'facet',
    14: 'crust',
    15: 'snowdrift',
    16: 'ground-facet',
    18: 'crust-above-facet ',
    19: 'crust-below-facet',
    20: 'ground-water',
    22: 'water-layers',
    24: 'loose',
    25: 'rain-temp-sun',
}

AVALANCHE_EXT = {
    10: "dry_loose",
    15: "wet_loose",
    20: "dry_slab",
    25: "wet_slab",
    27: "glide",
    30: "slush",
    40: "cornice",
}

EXPOSED_HEIGHTS = {
    1: "bottom-white",
    2: "bottom-black",
    3: "middle-white",
    4: "middle-black",
}

REG_ENG = {
    "Faretegn": "dangersign",
    "Tester": "tests",
    "Skredaktivitet": "activity",
    "Skredhendelse": "event",
    "Snødekke": "snowpack",
    "Skredproblem": "problem",
    "Skredfarevurdering": "danger",
    "Snøprofil": "snowprofile",
}

REGOBS_CLASSES = {
    "Faretegn": {
        "DangerSignTID": {
            2: 'avalanches',
            3: 'whumpf',
            4: 'cracks',
            5: 'snowfall',
            6: 'hoar',
            7: 'temp',
            8: 'water',
            9: 'snowdrift',
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
    },
    "Skredaktivitet": {
        "AvalancheExtTID": AVALANCHE_EXT,
        "ExposedHeightComboTID": EXPOSED_HEIGHTS,
    },
    "Skredhendelse": {
        "AvalancheTID": {
            11: "wet_loose",
            12: "dry_loose",
            21: "wet_slab",
            22: "dry_slab",
            27: "glide",
            30: "slush",
            40: "cornice",
        },
        "AvalancheTriggerTID": {
            10: "natural",
            20: "artificial",
            21: "artificial-skier",
            22: "remote",
            23: "artificial-test",
            25: "explosive",
            26: "human",
            27: "snowmobile",
        },
        "TerrainStartZoneTID": {
            10: "steep",
            20: "lee",
            30: "ridge",
            40: "gully",
            50: "slab",
            60: "bowl",
            70: "forest",
            75: "logging",
            95: "everywhere",
        },
        "AvalCauseTID": CAUSES,
    },
    "Snødekke": {
        "SnowSurfaceTID": {
            50: "facet",
            61: "hard_hoar",
            62: "soft_hoar",
            101: "max_loose",
            102: "med_loose",
            103: "min_loose",
            104: "wet_loose",
            105: "hard_wind",
            106: "soft_wind",
            107: "crust",
        }
    },
    "Skredproblem": {
        "AvalCauseTID": CAUSES,
        "AvalancheExtTID": AVALANCHE_EXT,
        "ExposedHeightComboTID": EXPOSED_HEIGHTS,
    },
    "Skredfarevurdering": {},
    "Snøprofil": {}
}

REGOBS_SCALARS = {
    "Faretegn": {},
    "Tester": {
        "FractureDepth": ("FractureDepth", lambda x: x),
        "TapsFracture": ("TapsFracture", lambda x: x),
        "StabilityEval": ("StabilityEvalTID", lambda x: x),
        "ComprTestFracture": ("ComprTestFractureTID", lambda x: x),
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
    "Skredhendelse": {
        "DestructiveSize": ("DestructiveSizeTID", lambda x: x if 0 < x <= 5 else 0),
        "FractureHeight": ("FractureHeigth", lambda x: x),  # sic
        "FractureWidth": ("FractureWidth", lambda x: x),
        "HeightStartZone": ("HeigthStartZone", lambda x: x),  # sic
        "HeightStopZone": ("HeigthStopZone", lambda x: x),  # sic
        "ValidExpositionN": ("ValidExposition", lambda x: float(x[0])),
        "ValidExpositionNE": ("ValidExposition", lambda x: float(x[1])),
        "ValidExpositionE": ("ValidExposition", lambda x: float(x[2])),
        "ValidExpositionSE": ("ValidExposition", lambda x: float(x[3])),
        "ValidExpositionS": ("ValidExposition", lambda x: float(x[4])),
        "ValidExpositionSW": ("ValidExposition", lambda x: float(x[5])),
        "ValidExpositionW": ("ValidExposition", lambda x: float(x[6])),
        "ValidExpositionNW": ("ValidExposition", lambda x: float(x[7])),
    },
    "Snødekke": {
        "SnowDepth": ("SnowDepth", lambda x: x),
        "NewSnowDepth24": ("NewSnowDepth24", lambda x: x),
        "Snowline": ("Snowline", lambda x: x),
        "NewSnowline": ("NewSnowLine", lambda x: x),
        "HeightLimitLayeredSnow": ("HeightLimitLayeredSnow", lambda x: x),
        "SnowDrift": ("SnowDriftTID", lambda x: x),
        "SurfaceWaterContent": ("SurfaceWaterContentTID", lambda x: x),
    },
    "Skredproblem": {
        "AvalCauseDepth": ("AvalCauseDepthTID", lambda x: x),
        "AvalCauseLight": ("AvalCauseAttributeLightTID", lambda x: float(bool(x))),
        "AvalCauseThin": ("AvalCauseAttributeThinTID", lambda x: float(bool(x))),
        "AvalCauseSoft": ("AvalCauseAttributeSoftTID", lambda x: float(bool(x))),
        "AvalCauseCrystal": ("AvalCauseAttributeCrystalTID", lambda x: float(bool(x))),
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
    "Skredfarevurdering": {
        "AvalancheDanger": ("AvalancheDangerTID", lambda x: x),
        "ForecastCorrect": ("ForecastCorrectTID", lambda x: x),
    },
    "Snøprofil": {
        "DH": ("StratProfile", lambda x: float("DH" in [y["GrainFormPrimaryTName"][:2] for y in x["Layers"]])),
        "SH": ("StratProfile", lambda x: float("SH" in [y["GrainFormPrimaryTName"][:2] for y in x["Layers"]])),
        "FC": ("StratProfile", lambda x: float("FC" in [y["GrainFormPrimaryTName"][:2] for y in x["Layers"]])),
        "H_PWL_F": (
            "StratProfile",
            lambda x: float(True in [
                _pwl.match(y["GrainFormPrimaryTName"]) and "F" == y["HardnessTName"][0] for y in x["Layers"]
            ]),
        ),
        "H_PWL_4F": (
            "StratProfile",
            lambda x: float(True in [
                _pwl.match(y["GrainFormPrimaryTName"]) and "4F" in y["HardnessTName"] for y in x["Layers"]
            ]),
        ),
        "H_F": ("StratProfile", lambda x: float(True in ["F" == y["HardnessTName"][0] for y in x["Layers"]])),
        "H_4F": ("StratProfile", lambda x: float(True in ["4F" in y["HardnessTName"] for y in x["Layers"]])),
        "W_M": ("StratProfile", lambda x: float(True in ["M" in y["WetnessTName"] for y in x["Layers"]])),
        "W_W": ("StratProfile", lambda x: float(True in ["W" in y["WetnessTName"] for y in x["Layers"]])),
        "W_V": ("StratProfile", lambda x: float(True in ["V" in y["WetnessTName"] for y in x["Layers"]])),
        "W_S": ("StratProfile", lambda x: float(True in ["S" in y["WetnessTName"] for y in x["Layers"]])),
        "T_max": ("SnowTemp", lambda x: max([y["SnowTemp"] for y in x["Layers"]], default=0)),
        "T_mean": ("SnowTemp", lambda x: sum([y["SnowTemp"] for y in x["Layers"]]) / (len(x["Layers"]) or 1)),
        "T_min": ("SnowTemp", lambda x: min([y["SnowTemp"] for y in x["Layers"]], default=0)),
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

        aw = []
        raw_regobs = {}
        for season in seasons:
            aw += gvp.get_all_forecasts(year=season)
            regions = gm.get_forecast_regions(year=season, get_b_regions=True)
            raw_regobs = {**raw_regobs, **_get_regobs_obs(regions, season, regobs_types)}

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

            label = OrderedDict({})
            label[('danger_level', 'CLASS')] = forecast.danger_level
            label[('emergency_warning', 'CLASS')] = forecast.emergency_warning

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
            for wind_dir in DIRECTIONS:
                weather[f"wind_dir_{wind_dir}"] = float(forecast.mountain_weather.wind_direction == wind_dir)
            for wind_dir in DIRECTIONS:
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
            label[('problem_amount', 'CLASS')] = prb['problem_amount']
            for i in range(1, 4):
                label[(f"problem_{i}", "CLASS")] = "none"
            for problem in PROBLEMS.values():
                if problem in problem_types:
                    index = problem_types.index(problem)
                    problems[problem] = forecast.avalanche_problems[index]
                    prb[f"problem_{problem}"] = -(problems[problem].avalanche_problem_id - 4)
                    label[(f"problem_{index}", "CLASS")] = problem
                else:
                    problems[problem] = gf.AvalancheWarningProblem()
                    prb[f"problem_{problem}"] = 0
            for problem in PROBLEMS.values():
                p_data = problems[problem]
                forecast_cause = CAUSES.get(p_data.aval_cause_id, None)
                for cause in CAUSES.values():
                    prb[f"problem_{problem}_cause_{cause}"] = float(forecast_cause == cause)
                prb[f"problem_{problem}_dsize"] = p_data.destructive_size_ext_id
                prb[f"problem_{problem}_prob"] = p_data.aval_probability_id
                prb[f"problem_{problem}_trig"] = {10: 0, 21: 1, 22: 2}.get(p_data.aval_trigger_simple_id, 0)
                prb[f"problem_{problem}_dist"] = p_data.aval_distribution_id
                prb[f"problem_{problem}_lev_max"] = p_data.exposed_height_1
                prb[f"problem_{problem}_lev_min"] = p_data.exposed_height_2

                label[(f"problem_{problem}_cause", 'CLASS')] = forecast_cause
                label[(f"problem_{problem}_dsize", 'CLASS')] = p_data.destructive_size_ext_id
                label[(f"problem_{problem}_prob", 'CLASS')] = p_data.aval_probability_id
                label[(f"problem_{problem}_trig", 'CLASS')] = p_data.aval_trigger_simple_id
                label[(f"problem_{problem}_dist", 'CLASS')] = p_data.aval_distribution_id
                label[(f"problem_{problem}_lev_fill", 'CLASS')] = p_data.exposed_height_fill

                for n in range(1, 5):
                    prb[f"problem_{problem}_lev_fill{n}"] = float(p_data.exposed_height_fill == n)
                for n in range(0, 8):
                    aspect_attr_name = f"problem_{problem}_aspect_{DIRECTIONS[n]}"
                    prb[aspect_attr_name] = float(p_data.valid_expositions[n])
                    label[(aspect_attr_name, 'CLASS')] = prb[aspect_attr_name]

                label[(f"problem_{problem}_lev_max", 'REAL')] = p_data.exposed_height_1
                label[(f"problem_{problem}_lev_min", 'REAL')] = p_data.exposed_height_2

                # Check for consistency
                if prb[f"problem_{problem}_lev_min"] > prb[f"problem_{problem}_lev_max"]:
                    continue

            row['problems'] = prb
            row['label'] = label

            # RegObs data
            row['regobs'] = {}
            current_regobs = raw_regobs.get((forecast.region_id, forecast.date_valid), {})
            # Use 5 most competent observations, and list both categories as well as scalars
            for obs_idx in range(0, 5):
                # One type of observation (test, danger signs etc.) at a time
                for regobs_type in self.regobs_types:
                    obses = current_regobs[regobs_type] if regobs_type in current_regobs else []
                    # Go through each requested class attribute from the specified observation type
                    for attr, cat in REGOBS_CLASSES[regobs_type].items():
                        # We handle categories using 1-hot, so we step through each category
                        for cat_name in cat.values():
                            attr_name = f"regobs_{REG_ENG[regobs_type]}_{_camel_to_snake(attr)}_{cat_name}_{obs_idx}"
                            row['regobs'][attr_name] = obses[obs_idx][cat_name] if len(obses) > obs_idx else 0
                    # Go through all requested scalars
                    for attr, (regobs_attr, conv) in REGOBS_SCALARS[regobs_type].items():
                        attr_name = f"regobs_{REG_ENG[regobs_type]}_{_camel_to_snake(attr)}_{obs_idx}"
                        try:
                            row['regobs'][attr_name] = conv(obses[obs_idx][regobs_attr]) if len(obses) > obs_idx else 0
                        except TypeError:
                            row['regobs'][attr_name] = 0

            # Check for consistency
            if weather['temp_min'] > weather['temp_max']:
                continue

            self.tree[(forecast.region_id, forecast.date_valid)] = row

    def label(self, days):
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

        :return:                LabeledData
        """
        label_table = OrderedDict({})
        table = OrderedDict({})
        df = None
        df_label = None
        days_w = {0: 1, 1: 1, 2: 1}.get(days, days - 1)
        days_v = {0: 1, 1: 2, 2: 2}.get(days, days)
        days_r = days + 1

        for entry_idx, entry in enumerate(self.tree.values()):
            def prev(day_dist):
                return self.tree[(entry['region_id'], entry['date'] - dt.timedelta(days=day_dist))]
            # Skip B-regions for now.
            if entry['region_type'] == 'B':
                continue
            try:
                for n in range(0, days_r):
                    # Just check that we can use this day.
                    prev(n)
            except KeyError:
                continue

            row = OrderedDict({})
            for region in REGIONS:
                row[(f"region_id_{region}", 0)] = float(region == entry["region_id"])

            # It would obviously be better code-wise to flip the loops, but we need this insertion order.
            for key in entry['weather'].keys():
                for n in range(0, days_w):
                    row[(key, n)] = prev(n)['weather'][key]
            for n in range(1, days_v):
                row[("danger_level", n)] = prev(n)['danger_level']
            for n in range(1, days_v):
                row[("emergency_warning", n)] = prev(n)['emergency_warning']
            for key in entry['problems'].keys():
                for n in range(1, days_v):
                    row[(key, n)] = prev(n)['problems'][key]
            for key in entry['regobs'].keys():
                for n in range(2, days_r):
                    row[(key, n)] = prev(n)['regobs'][key]

            for datum, data in [(row, table), (entry['label'], label_table)]:
                # Some restructuring to make DataFrame parse the dict correctly
                for key in datum.keys():
                    if key not in data:
                        data[key] = OrderedDict({})
                    data[key][entry_idx] = datum[key]
            # Build DataFrame iteratively to preserve system memory (floats in dicts are apparently expensive).
            if entry_idx % 1000 == 0:
                df_new = pandas.DataFrame(table).astype(np.float32).fillna(0)
                df_label_new = pandas.DataFrame(label_table)
                df = df_new if df is None else pandas.concat([df, df_new], ignore_index=True)
                df_label = df_label_new if df is None else pandas.concat([df_label, df_label_new], ignore_index=True)
                table = OrderedDict({})
                label_table = OrderedDict({})

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

    def to_csv(self):
        """ Writes a csv-file in `varsomdata/localstorage` named according to the properties of the dataset.
        A `label.csv` is also always written.
        """
        # Write training data
        regobs = ""
        if len(self.regobs_types) and self.days >= 2:
            regobs = f"_regobs_{'_'.join([REG_ENG[obs_type] for obs_type in self.regobs_types])}"
        pathname_data = f"{se.local_storage}data_days_{self.days}{regobs}.csv"
        pathname_label = f"{se.local_storage}label_days_{self.days}{regobs}.csv"
        ld = self.denormalize()
        ld.data.to_csv(pathname_data, sep=';')
        ld.label.to_csv(pathname_label, sep=';')

    def copy(self):
        """Deep copy LabeledData."""
        ld = LabeledData(self.data.copy(), self.label.copy(), self.days, copy.copy(self.regobs_types))
        ld.scaler = self.scaler
        return ld

    @staticmethod
    def from_csv(days, regobs_types):
        """Read LabeledData from previously written .csv-file.

        :param days:            How far back in time values should data be included.
        :param regobs_types:    A tuple/list of strings of types of observations to fetch from RegObs.,
                                e.g., `("Faretegn")`.
        """
        regobs = ""
        if len(regobs_types) and days >= 2:
            regobs = f"_regobs_{'_'.join([REG_ENG[obs_type] for obs_type in regobs_types])}"
        pathname_data = f"{se.local_storage}data_days_{days}{regobs}.csv"
        pathname_label = f"{se.local_storage}label_days_{days}{regobs}.csv"
        try:
            data = pandas.read_csv(pathname_data, sep=";", header=[0, 1], index_col=0)
            label = pandas.read_csv(pathname_label, sep=";", header=[0, 1], index_col=0, low_memory=False)
        except FileNotFoundError:
            raise CsvMissingError()
        return LabeledData(data, label, days, regobs_types)


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
    # Make sure all requested elements from RegObs actually have the information we need specified
    if not min(map(lambda x: set(list(x.keys())).issuperset(req_set), [REGOBS_CLASSES, REGOBS_SCALARS, REG_ENG])):
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
    else:
        with open(file_name, 'rb') as handle:
            response = pickle.load(handle)

    for raw_obs in response:
        for reg in raw_obs["Registrations"]:
            obs_type = reg["RegistrationName"]
            if obs_type not in requested_types:
                continue
            # Ignore snow profiles of the old format
            if obs_type == "Snøprofil" and "StratProfile" not in reg["FullObject"]:
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


class CsvMissingError(Error):
    pass


if __name__ == '__main__':
    forecast_dataset = ForecastDataset(regobs_types=list(REG_ENG.keys()))
    for n in range(0, 8):
        forecast_dataset.label(days=n).to_csv()
