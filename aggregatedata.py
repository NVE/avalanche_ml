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
from sklearn.model_selection import KFold


old_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "./varsomdata")
import setenvironment as se
from varsomdata import getforecastapi as gf
from varsomdata import getvarsompickles as gvp
from varsomdata import getmisc as gm
os.chdir(old_dir)


__author__ = 'arwi'

_pwl = re.compile("(DH|SH|FC)")

CSV_VERSION = "19"

_NONE = ""

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
    # This was safe to do when it was written (rewriting dpwl as pwl).
    # If reverse lookups are done in the future, or if an .items() iteration
    # is done, this may break something.
    37: 'pwl-slab',
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
    18: 'crust-above-facet',
    19: 'crust-below-facet',
    20: 'ground-water',
    22: 'water-layers',
    24: 'loose',
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
            label[('CLASS', _NONE, 'danger_level')] = forecast.danger_level
            label[('CLASS', _NONE, 'emergency_warning')] = forecast.emergency_warning

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
            problem_types = [PROBLEMS.get(p.avalanche_problem_type_id, _NONE) for p in forecast.avalanche_problems]
            problems = {}
            prb['problem_amount'] = len(forecast.avalanche_problems)
            label[('CLASS', _NONE, 'problem_amount')] = prb['problem_amount']
            for i in range(1, 4):
                label[('CLASS', _NONE, f"problem_{i}")] = _NONE
            for problem in PROBLEMS.values():
                if problem in problem_types:
                    index = problem_types.index(problem)
                    problems[problem] = forecast.avalanche_problems[index]
                    prb[f"problem_{problem}"] = -(problems[problem].avalanche_problem_id - 4)
                    label[('CLASS', _NONE, f"problem_{index + 1}")] = problem
                else:
                    problems[problem] = gf.AvalancheWarningProblem()
                    prb[f"problem_{problem}"] = 0
            for problem in PROBLEMS.values():
                p_data = problems[problem]
                forecast_cause = CAUSES.get(p_data.aval_cause_id, _NONE)
                for cause in CAUSES.values():
                    prb[f"problem_{problem}_cause_{cause}"] = float(forecast_cause == cause)
                prb[f"problem_{problem}_dsize"] = p_data.destructive_size_ext_id
                prb[f"problem_{problem}_prob"] = p_data.aval_probability_id
                prb[f"problem_{problem}_trig"] = {10: 0, 21: 1, 22: 2}.get(p_data.aval_trigger_simple_id, 0)
                prb[f"problem_{problem}_dist"] = p_data.aval_distribution_id
                prb[f"problem_{problem}_lev_max"] = p_data.exposed_height_1
                prb[f"problem_{problem}_lev_min"] = p_data.exposed_height_2

                label[('CLASS', problem, "cause")] = forecast_cause
                label[('CLASS', problem, "dsize")] = p_data.destructive_size_ext_id
                label[('CLASS', problem, "prob")] = p_data.aval_probability_id
                label[('CLASS', problem, "trig")] = p_data.aval_trigger_simple_id
                label[('CLASS', problem, "dist")] = p_data.aval_distribution_id
                label[('CLASS', problem, "lev_fill")] = p_data.exposed_height_fill

                for n in range(1, 5):
                    prb[f"problem_{problem}_lev_fill{n}"] = float(p_data.exposed_height_fill == n)
                for n in range(0, 8):
                    aspect_attr_name = f"problem_{problem}_aspect_{DIRECTIONS[n]}"
                    prb[aspect_attr_name] = float(p_data.valid_expositions[n])
                label[('MULTI', problem, "aspect")] = p_data.valid_expositions.zfill(8)
                label[('REAL', problem, "lev_max")] = p_data.exposed_height_1
                label[('REAL', problem, "lev_min")] = p_data.exposed_height_2

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

            row['accuracy'] = map(
                lambda x: {0: 0, 1: 1, 2: -1, 3: -1}[x['ForecastCorrectTID']],
                current_regobs['Skredfarevurdering']
            ) if 'Skredfarevurdering' in current_regobs else []

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
        row_weight = OrderedDict({})
        df = None
        df_label = None
        df_weight = None
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

            weight_sum = sum(entry['accuracy'])
            if weight_sum < 0:
                row_weight[(entry['date'].isoformat(), entry['region_id'])] = 1 / 2
            elif weight_sum == 0:
                row_weight[(entry['date'].isoformat(), entry['region_id'])] = 1
            elif weight_sum > 0:
                row_weight[(entry['date'].isoformat(), entry['region_id'])] = 2

            for datum, data in [(row, table), (entry['label'], label_table)]:
                # Some restructuring to make DataFrame parse the dict correctly
                for key in datum.keys():
                    if key not in data:
                        data[key] = OrderedDict({})
                    data[key][(entry['date'].isoformat(), entry['region_id'])] = datum[key]
            # Build DataFrame iteratively to preserve system memory (floats in dicts are apparently expensive).
            if entry_idx % 1000 == 0:
                df_new = pandas.DataFrame(table, dtype=np.float32).fillna(0)
                df_label_new = pandas.DataFrame(label_table, dtype="U")
                df_weight_new = pandas.Series(row_weight)
                df = df_new if df is None else pandas.concat([df, df_new])
                df_label = df_label_new if df is None else pandas.concat([df_label, df_label_new])
                df_weight = df_weight_new if df is None else pandas.concat([df_weight, df_weight_new])
                table = OrderedDict({})
                label_table = OrderedDict({})
                row_weight = OrderedDict({})

        return LabeledData(df, df_label, df_weight, days, self.regobs_types)


class LabeledData:
    is_normalized = False
    scaler = MinMaxScaler()

    def __init__(self, data, label, row_weight, days, regobs_types):
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
        """
        self.data = data
        self.row_weight = row_weight
        self.label = label.sort_index(axis=1)
        self.label = self.label.replace(_NONE, 0)
        self.label = self.label.replace(np.nan, 0)
        self.label['CLASS', _NONE] = self.label['CLASS', _NONE].replace(0, _NONE).values
        self.label['MULTI'] = self.label['MULTI'].replace(0, "0").values
        self.pred = label.copy()
        for col in self.pred.columns:
            self.pred[col].values[:] = 0
        self.pred['CLASS', _NONE] = _NONE
        self.pred['MULTI'] = "0"
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

    def kfold(self, k=5, shuffle=True):
        """Returns an iterable of LabeledData-tuples. The first element is the training dataset
        and the second is for testing.

        :param k: Int: Number of folds.
        :param shuffle: Bool: Whether rows should be shuffled before folding. Defaults to True.
        :return: Iterable<(LabeledData, LabeledData)>
        """
        kf = KFold(n_splits=k, shuffle=shuffle)
        array = []
        for train_index, test_index in kf.split(self.data):
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
        df = None
        problems = [(label, "CLASS") for label in dummies["label"]["CLASS"].keys()]
        problems += [(label, "MULTI") for label in dummies["label"]["MULTI"].keys()]
        old_settings = np.seterr(divide='raise', invalid='raise')
        for subprob, typ in problems:
            truth = dummies["label"][typ][subprob]
            pred = dummies["pred"][typ][subprob]
            true_pos = np.sum(truth * pred, axis=0)
            try:
                prec = true_pos / np.sum(pred, axis=0)
            except FloatingPointError:
                prec = pandas.Series(index=pred.columns).fillna(0)
            try:
                recall = true_pos / np.sum(truth, axis=0)
            except FloatingPointError:
                recall = pandas.Series(index=pred.columns).fillna(0)
            try:
                f1 = 2 * prec * recall / (prec + recall)
            except FloatingPointError:
                f1 = pandas.Series(index=pred.columns).fillna(0)
            new_df = pandas.DataFrame(index=truth.columns, columns=['f1', 'precision', 'recall', 'rmse'])
            new_df.iloc[:, :3] = np.array([f1, prec, recall]).transpose()
            df = new_df if df is None else pandas.concat([df, new_df], sort=True)

        for subprob in dummies["label"]["REAL"].keys():
            truth = dummies["label"]["REAL"][subprob].values
            pred = dummies["pred"]["REAL"][subprob].values
            try:
                rmse = (np.sqrt(np.sum(np.square(pred - truth), axis=0)) / truth.shape[0])
            except:
                rmse = 0
            new_df = pandas.DataFrame({"rmse": rmse}, index=dummies["label"]["REAL"][subprob].columns)
            df = new_df if df is None else pandas.concat([df, new_df], sort=True)
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

        :return: dict<dict<dict<pandas.DataFrame>>> (In the future this will be a flat DataFrame)
        """
        dummies = {}
        for name, df in [('label', self.label), ('pred', self.pred)]:
            dummies[name] = {"CLASS": {}, "MULTI": {}, "REAL": {}}
            for subprob in df["CLASS"].columns.get_level_values(0).unique():
                if name == 'label':
                    if subprob == _NONE:
                        sub_df = df["CLASS"][subprob]
                    else:
                        sub_df = df["CLASS"][subprob].loc[
                            df["CLASS"][_NONE]["problem_1"].eq(subprob).astype(np.int) +
                            df["CLASS"][_NONE]["problem_2"].eq(subprob).astype(np.int) +
                            df["CLASS"][_NONE]["problem_3"].eq(subprob).astype(np.int) > 0
                        ]
                    col = pandas.get_dummies(sub_df, prefix_sep=':').columns
                    dum = pandas.DataFrame(pandas.get_dummies(df["CLASS"][subprob], prefix_sep=':'), columns=col)
                    dummies[name]["CLASS"][subprob] = dum
                else:
                    col = dummies["label"]["CLASS"][subprob].columns
                    dum = pandas.DataFrame(pandas.get_dummies(df["CLASS"][subprob], prefix_sep=':'), columns=col)

                    dummies[name]["CLASS"][subprob] = dum

                    columns = dummies["label"]["CLASS"][subprob].columns.values.astype("U")
                    idx = pandas.MultiIndex.from_tuples(
                        [("CLASS", subprob, a[0], a[2]) for a in np.char.partition(columns, sep=":")],
                        names=["type", "problem", "attribute", "label"]
                    )
                    dummies["label"]["CLASS"][subprob].columns = idx
                    dummies["pred"]["CLASS"][subprob].columns = idx

            for subprob in df['MULTI'].columns.get_level_values(0).unique():
                multi = df['MULTI'][subprob].replace(_NONE, "0").values.astype(np.int).astype("U")
                if name == 'label':
                    multimax = np.max(np.char.str_len(multi), axis=0)
                multi = np.char.zfill(multi, multimax)
                multi = np.nan_to_num(np.array([[list(elem) for elem in row] for row in multi]))
                multi = multi.reshape(multi.shape[0], multi.shape[1] * multi.shape[2]).astype(np.float)
                columns = zip(df["MULTI"][subprob].columns, multimax)
                columns = [[("MULTI", subprob, c, str(n)) for n in range(max)] for c, max in columns]
                columns = [item for sublist in columns for item in sublist]
                columns = pandas.MultiIndex.from_tuples(columns, names=["type", "problem", "attribute", "label"])
                dummies[name]["MULTI"][subprob] = pandas.DataFrame(multi, index=df.index, columns=columns)

            for subprob in df["REAL"].columns.get_level_values(0).unique():
                columns = pandas.MultiIndex.from_tuples(
                    [("REAL", subprob, a, "") for a in df["REAL"][subprob].columns],
                    names=["type", "problem", "attribute", "label"]
                )
                dummies[name]["REAL"][subprob] = pandas.DataFrame(
                    df['REAL'][subprob].values,
                    columns=columns,
                    index=df.index
                )

        return dummies

    def to_csv(self):
        """ Writes a csv-file in `varsomdata/localstorage` named according to the properties of the dataset.
        A `label.csv` is also always written.
        """
        # Write training data
        regobs = ""
        if len(self.regobs_types) and self.days >= 2:
            regobs = f"_regobs_{'_'.join([REG_ENG[obs_type] for obs_type in self.regobs_types])}"
        pathname_data = f"{se.local_storage}data_v{CSV_VERSION}_days_{self.days}{regobs}.csv"
        pathname_label = f"{se.local_storage}label_v{CSV_VERSION}_days_{self.days}{regobs}.csv"
        pathname_weight = f"{se.local_storage}weight_v{CSV_VERSION}_days_{self.days}{regobs}.csv"
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
                    ext_val = dict['values'][row['CLASS', '', int_attr]][idx]
                    setattr(aw, ext_attr, ext_val)
            for p_idx in range(1, int(row['CLASS', '', 'problem_amount']) + 1):
                p_prefix = f"problem_{p_idx}"
                p_name = row['CLASS', '', p_prefix]
                if p_name == "":
                    break
                problem = gf.AvalancheWarningProblem()
                problem.avalanche_problem_id = -p_idx + 4
                for idx, ext_attr in enumerate(LABEL_PROBLEM_PRIMARY['ext_attr']):
                    ext_val = LABEL_PROBLEM_PRIMARY['values'][row['CLASS', '', p_prefix]][idx]
                    setattr(problem, ext_attr, ext_val)
                for int_attr, dict in LABEL_PROBLEM.items():
                    for idx, ext_attr in enumerate(dict['ext_attr']):
                        ext_val = dict['values'][row['CLASS', p_name, int_attr]][idx]
                        setattr(problem, ext_attr, ext_val)
                for int_attr, dict in LABEL_PROBLEM_MULTI.items():
                    ext_attr = dict['ext_attr']
                    ext_val = row['MULTI', p_name, int_attr]
                    setattr(problem, ext_attr, ext_val)
                for int_attr, dict in LABEL_PROBLEM_REAL.items():
                    ext_attr = dict['ext_attr']
                    ext_val = row['REAL', p_name, int_attr]
                    setattr(problem, ext_attr, ext_val)
                aw.avalanche_problems.append(problem)
            aws.append(aw)
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
            copy.copy(self.regobs_types)
        )
        ld.is_normalized = self.is_normalized
        ld.scaler = self.scaler
        ld.pred = self.pred.copy(deep=True)
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
        pathname_data = f"{se.local_storage}data_v{CSV_VERSION}_days_{days}{regobs}.csv"
        pathname_label = f"{se.local_storage}label_v{CSV_VERSION}_days_{days}{regobs}.csv"
        pathname_weight = f"{se.local_storage}weight_v{CSV_VERSION}_days_{days}{regobs}.csv"
        try:
            data = pandas.read_csv(pathname_data, sep=";", header=[0, 1], index_col=[0, 1])
            label = pandas.read_csv(pathname_label, sep=";", header=[0, 1, 2], index_col=[0, 1], low_memory=False, dtype="U")
            row_weight = pandas.read_csv(pathname_weight, sep=";", header=None, index_col=[0, 1], low_memory=False)
            columns = [(col[0], re.sub(r'Unnamed:.*', _NONE, col[1]), col[2]) for col in label.columns.tolist()]
            label.columns = pandas.MultiIndex.from_tuples(columns)
        except FileNotFoundError:
            raise CsvMissingError()
        return LabeledData(data, label, row_weight, days, regobs_types)


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
        try:
            with open(file_name, 'rb') as handle:
                response = pickle.load(handle)
        except:
            os.remove(file_name)
            return _get_regobs_obs(regions, year, requested_types, max_file_age)

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
