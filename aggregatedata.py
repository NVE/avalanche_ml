# -*- coding: utf-8 -*-
"""Structures data in ML-friendly ways."""

import copy
import datetime as dt
import os
import pickle
import re
import sys
import math
from concurrent import futures
from collections import OrderedDict
import numpy as np
import pandas
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold


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

CSV_VERSION = "25"

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

# Transformations from Varsom main level
AVALANCHE_WARNING = {
    "danger_level": ("danger_level", lambda x: x),
    "emergency_warning": ("emergency_warning", lambda x: float(x == "Ikke gitt")),
    "problem_amount": ("avalanche_problems", lambda x: len(x)),
}

# Same as AVALANCHE_WARNING, but destined for the label table
AVALANCHE_WARNING_LABEL = {
    ("CLASS", "danger_level"): ("danger_level", lambda x: x),
    ("CLASS", "emergency_warning"): ("emergency_warning", lambda x: x),
    ("CLASS", "problem_amount"): ("avalanche_problems", lambda x: len(x)),
}

# Transformations from Varsom problem level
AVALANCHE_PROBLEM = {
    "dsize": ("destructive_size_ext_id", lambda x: x),
    "prob": ("aval_probability_id", lambda x: x),
    "trig": ("aval_trigger_simple_id", lambda x: {10: 0, 21: 1, 22: 2}.get(x, 0)),
    "dist": ("aval_distribution_id", lambda x: x),
    "lev_max": ("exposed_height_1", lambda x: x),
    "lev_min": ("exposed_height_2", lambda x: x),
    "cause_new-snow": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'new-snow')),
    "cause_hoar": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_hoar')),
    "cause_facet": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_facet')),
    "cause_crust": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_crust')),
    "cause_snowdrift": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_snowdrift')),
    "cause_ground-facet": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_ground-facet')),
    "cause_crust-above-facet": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_crust-above-facet')),
    "cause_crust-below-facet": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_crust-below-facet')),
    "cause_ground-water": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_ground-water')),
    "cause_water-layers": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_water-layers')),
    "cause_loose": ("aval_cause_id", lambda x: float(CAUSES.get(x, _NONE) == 'cause_loose')),
    "lev_fill_1": ("exposed_height_fill", lambda x: float(x == 1)),
    "lev_fill_2": ("exposed_height_fill", lambda x: float(x == 2)),
    "lev_fill_3": ("exposed_height_fill", lambda x: float(x == 3)),
    "lev_fill_4": ("exposed_height_fill", lambda x: float(x == 4)),
    "aspect_N": ("valid_expositions", lambda x: float(x[0])),
    "aspect_NE": ("valid_expositions", lambda x: float(x[1])),
    "aspect_E": ("valid_expositions", lambda x: float(x[2])),
    "aspect_SE": ("valid_expositions", lambda x: float(x[3])),
    "aspect_S": ("valid_expositions", lambda x: float(x[4])),
    "aspect_SW": ("valid_expositions", lambda x: float(x[5])),
    "aspect_W": ("valid_expositions", lambda x: float(x[6])),
    "aspect_NW": ("valid_expositions", lambda x: float(x[7])),
}

# Same as AVALANCHE_PROBLEM, but destined for the label table
AVALANCHE_PROBLEM_LABEL = {
    ("CLASS", "cause"): ("aval_cause_id", lambda x: CAUSES.get(x, _NONE)),
    ("CLASS", "dsize"): ("destructive_size_ext_id", lambda x: x),
    ("CLASS", "prob"): ("aval_probability_id", lambda x: x),
    ("CLASS", "trig"): ("aval_trigger_simple_id", lambda x: x),
    ("CLASS", "dist"): ("aval_distribution_id", lambda x: x),
    ("CLASS", "lev_fill"): ("exposed_height_fill", lambda x: x),
    ("MULTI", "aspect"): ("valid_expositions", lambda x: x.zfill(8)),
    ("REAL", "lev_max"): ("exposed_height_1", lambda x: x),
    ("REAL", "lev_min"): ("exposed_height_2", lambda x: x),
}

# Transformations from Mountain Weather API
WEATHER_VARSOM = {
    "precip_most_exposed": ("precip_most_exposed", lambda x: x),
    "precip": ("precip_region", lambda x: x),
    "wind_speed": ("wind_speed", lambda x: WIND_SPEEDS.get(x, 0)),
    "wind_change_speed": ("change_wind_speed", lambda x: WIND_SPEEDS.get(x, 0)),
    "temp_min": ("temperature_min", lambda x: x),
    "temp_max": ("temperature_max", lambda x: x),
    "temp_lev": ("temperature_elevation", lambda x: x),
    "temp_freeze_lev": ("freezing_level", lambda x: x),
    "wind_dir_N": ("wind_direction", lambda x: x == "N"),
    "wind_dir_NE": ("wind_direction", lambda x: x == "NE"),
    "wind_dir_E": ("wind_direction", lambda x: x == "E"),
    "wind_dir_SE": ("wind_direction", lambda x: x == "SE"),
    "wind_dir_S": ("wind_direction", lambda x: x == "S"),
    "wind_dir_SW": ("wind_direction", lambda x: x == "SW"),
    "wind_dir_W": ("wind_direction", lambda x: x == "W"),
    "wind_dir_NW": ("wind_direction", lambda x: x == "NW"),
    "wind_chg_dir_N": ("change_wind_direction", lambda x: x == "N"),
    "wind_chg_dir_NE": ("change_wind_direction", lambda x: x == "NE"),
    "wind_chg_dir_E": ("change_wind_direction", lambda x: x == "E"),
    "wind_chg_dir_SE": ("change_wind_direction", lambda x: x == "SE"),
    "wind_chg_dir_S": ("change_wind_direction", lambda x: x == "S"),
    "wind_chg_dir_SW": ("change_wind_direction", lambda x: x == "SW"),
    "wind_chg_dir_W": ("change_wind_direction", lambda x: x == "W"),
    "wind_chg_dir_NW": ("change_wind_direction", lambda x: x == "NW"),
    "wind_chg_start_0": ("change_hour_of_day_start", lambda x: x == 0),
    "wind_chg_start_6": ("change_hour_of_day_start", lambda x: x == 6),
    "wind_chg_start_12": ("change_hour_of_day_start", lambda x: x == 12),
    "wind_chg_start_18": ("change_hour_of_day_start", lambda x: x == 18),
    "temp_fl_start_0": ("fl_hour_of_day_start", lambda x: x == 0),
    "temp_fl_start_6": ("fl_hour_of_day_start", lambda x: x == 6),
    "temp_fl_start_12": ("fl_hour_of_day_start", lambda x: x == 12),
    "temp_fl_start_18": ("fl_hour_of_day_start", lambda x: x == 18),
}

WEATHER_API = {
    "precip_most_exposed": ("Precipitation_MostExposed_Median", lambda x: x),
    "precip": ("Precipitation_overall_ThirdQuartile", lambda x: x),
    "wind_speed": ("WindClassification", lambda x: WIND_SPEEDS.get(x)),
    "temp_min": ("MinTemperature", lambda x: x),
    "temp_max": ("MaxTemperature", lambda x: x),
    "temp_lev": ("TemperatureElevation", lambda x: x),
    "temp_freeze_lev": ("FreezingLevelAltitude", lambda x: x),
    "wind_dir_N": ("WindDirection", lambda x: x == "N"),
    "wind_dir_NE": ("WindDirection", lambda x: x == "NE"),
    "wind_dir_E": ("WindDirection", lambda x: x == "E"),
    "wind_dir_SE": ("WindDirection", lambda x: x == "SE"),
    "wind_dir_S": ("WindDirection", lambda x: x == "S"),
    "wind_dir_SW": ("WindDirection", lambda x: x == "SW"),
    "wind_dir_W": ("WindDirection", lambda x: x == "W"),
    "wind_dir_NW": ("WindDirection", lambda x: x == "NW"),
    "temp_fl_start_0": ("FreezingLevelTime", lambda x: _round_hours(x) == 0),
    "temp_fl_start_6": ("FreezingLevelTime", lambda x: _round_hours(x) == 6),
    "temp_fl_start_12": ("FreezingLevelTime", lambda x: _round_hours(x) == 12),
    "temp_fl_start_18": ("FreezingLevelTime", lambda x: _round_hours(x) == 18),
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

# Transformations for RegObs Classes
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

# Transformations for RegObs scalars
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
        self.weather = {}
        self.regobs = {}
        self.varsom = {}
        self.labels = {}

        for season in seasons:
            regions = gm.get_forecast_regions(year=season, get_b_regions=True)
            varsom, labels = _get_varsom_obs(year=season)
            self.varsom = _merge(self.varsom, varsom)
            self.labels = _merge(self.labels, labels)
            regobs = _get_regobs_obs(regions, season, regobs_types)
            self.regobs = _merge(self.regobs, regobs)
            weather = _get_weather_obs(season)
            self.weather = _merge(self.weather, weather)

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
        df_label = pandas.DataFrame(self.labels, dtype="U")
        days_w = {0: 1, 1: 1, 2: 1}.get(days, days - 1)
        days_v = {0: 1, 1: 2, 2: 2}.get(days, days)
        days_r = days + 1
        varsom_index = pandas.DataFrame(self.varsom).index
        weather_index = pandas.DataFrame(self.weather).index

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

            row = OrderedDict({})
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
                df_new = pandas.DataFrame(table, dtype=np.float32).fillna(0)
                df_weight_new = pandas.Series(row_weight)
                df = df_new if df is None else pandas.concat([df, df_new])
                df_weight = df_weight_new if df is None else pandas.concat([df_weight, df_weight_new])
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
                prec = pandas.Series(index=pred.columns)
            try:
                recall = true_pos / np.sum(truth, axis=0)
            except FloatingPointError:
                recall = pandas.Series(index=pred.columns)
            try:
                f1 = 2 * prec * recall / (prec + recall)
            except FloatingPointError:
                f1 = pandas.Series(index=pred.columns)
            new_df = pandas.DataFrame(index=truth.columns, columns=['f1', 'precision', 'recall', 'rmse'])
            new_df.iloc[:, :3] = np.array([f1, prec, recall]).transpose()
            df = new_df if df is None else pandas.concat([df, new_df], sort=True)
            df[['f1', 'precision', 'recall']] = df[['f1', 'precision', 'recall']].fillna(0)

        for subprob in dummies["label"]["REAL"].keys():
            truth = dummies["label"]["REAL"][subprob].values
            pred = dummies["pred"]["REAL"][subprob].values
            try:
                ntruth = (truth - truth.min(axis=0)) / (truth.max(axis=0) - truth.min(axis=0))
                npred = (pred - pred.min(axis=0)) / (pred.max(axis=0) - pred.min(axis=0))
                rmse = (np.sqrt(np.sum(np.square(npred - ntruth), axis=0)) / ntruth.shape[0])
            except Exception:
                rmse = np.nan
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
            for subprob in df.loc[:, ["CLASS"]].columns.get_level_values(1).unique():
                try:
                    sub_df = self.label["CLASS", subprob]
                    try: col = pandas.get_dummies(sub_df, prefix_sep=':').columns
                    except ValueError: col = []

                    if name == 'label':
                        dum = pandas.DataFrame(pandas.get_dummies(sub_df, prefix_sep=':'), columns=col)
                        dummies[name]["CLASS"][subprob] = dum.fillna(0)

                        columns = dummies["label"]["CLASS"][subprob].columns.values.astype("U")
                        idx = pandas.MultiIndex.from_tuples(
                            [("CLASS", subprob, a[0], a[2]) for a in np.char.partition(columns, sep=":")],
                            names=["type", "problem", "attribute", "label"]
                        )
                        dummies["label"]["CLASS"][subprob].columns = idx
                    else:
                        dum = pandas.DataFrame(pandas.get_dummies(df["CLASS", subprob], prefix_sep=':'), columns=col)
                        dummies[name]["CLASS"][subprob] = dum.fillna(0)
                        dummies["pred"]["CLASS"][subprob].columns = dummies["label"]["CLASS"][subprob].columns
                except KeyError:
                    pass

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
                        columns = [[("MULTI", subprob, c, str(n)) for n in range(max)] for c, max in columns]
                        columns = [item for sublist in columns for item in sublist]
                        columns = pandas.MultiIndex.from_tuples(columns, names=["type", "problem", "attribute", "label"])
                        dummies[name]["MULTI"][subprob] = pandas.DataFrame(multi, index=df.index, columns=columns)
                    except KeyError:
                        pass
            except KeyError:
                pass

            try:
                for subprob in df.loc[:, ["REAL"]].columns.get_level_values(1).unique():
                    try:
                        columns = pandas.MultiIndex.from_tuples(
                            [("REAL", subprob, a, "") for a in df["REAL"][subprob].columns],
                            names=["type", "problem", "attribute", "label"]
                        )
                        dummies[name]["REAL"][subprob] = pandas.DataFrame(
                            df['REAL'][subprob].values,
                            columns=columns,
                            index=df.index
                        )
                    except KeyError:
                        pass
            except KeyError:
                pass

        return dummies

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
            data = pandas.read_csv(pathname_data, sep=";", header=[0, 1], index_col=[0, 1])
            label = pandas.read_csv(pathname_label, sep=";", header=[0, 1, 2], index_col=[0, 1], low_memory=False, dtype="U")
            row_weight = pandas.read_csv(pathname_weight, sep=";", header=None, index_col=[0, 1], low_memory=False, squeeze=True)
            columns = [(col[0], re.sub(r'Unnamed:.*', _NONE, col[1]), col[2]) for col in label.columns.tolist()]
            label.columns = pandas.MultiIndex.from_tuples(columns)
        except FileNotFoundError:
            raise CsvMissingError()
        return LabeledData(data, label, row_weight, days, regobs_types, with_varsom)

def _get_varsom_obs(year, max_file_age=23):
    aw = gvp.get_all_forecasts(year=year)
    forecasts = {}
    labels = {}
    for forecast in aw:
        # Skip B-regions for now.
        if forecast.region_type_name == 'B':
            continue

        row = {}
        table_row = {}

        for key, (orig_key, mapper) in AVALANCHE_WARNING.items():
            row[key] = mapper(getattr(forecast, orig_key))
        for (typ, attribute), (orig_key, mapper) in AVALANCHE_WARNING_LABEL.items():
            table_row[(typ, _NONE, attribute)] = mapper(getattr(forecast, orig_key))

        problem_types = [PROBLEMS.get(p.avalanche_problem_type_id, _NONE) for p in forecast.avalanche_problems]
        problems = {}
        for i in range(1, 4):
            table_row[('CLASS', _NONE, f"problem_{i}")] = _NONE
        for problem in PROBLEMS.values():
            if problem in problem_types:
                index = problem_types.index(problem)
                problems[problem] = forecast.avalanche_problems[index]
                row[f"problem_{problem}"] = -(problems[problem].avalanche_problem_id - 4)
                table_row[('CLASS', _NONE, f"problem_{index + 1}")] = problem
            else:
                problems[problem] = gf.AvalancheWarningProblem()
                row[f"problem_{problem}"] = 0

        for problem in PROBLEMS.values():
            p_data = problems[problem]
            for key, (orig_key, mapper) in AVALANCHE_PROBLEM.items():
                row[f"problem_{problem}_{key}"] = mapper(getattr(p_data, orig_key))
            for (type, attribute), (orig_key, mapper) in AVALANCHE_PROBLEM_LABEL.items():
                table_row[(type, problem, attribute)] = mapper(getattr(p_data, orig_key))

        for key, value in row.items():
            if key not in forecasts:
                forecasts[key] = {}
            forecasts[key][(forecast.date_valid.isoformat(), forecast.region_id)] = value
        for key, value in table_row.items():
            if key not in labels:
                labels[key] = {}
            labels[key][(forecast.date_valid.isoformat(), forecast.region_id)] = value
    return forecasts, labels

def _get_weather_obs(year, max_file_age=23):
    aw = gvp.get_all_forecasts(year=year)
    file_name = f'{se.local_storage}weather_v{CSV_VERSION}_{year}.pickle'
    file_date_limit = dt.datetime.now() - dt.timedelta(hours=max_file_age)
    current_season = gm.get_season_from_date(dt.date.today() - dt.timedelta(30))
    get_new = True

    try:
        # Don't fetch new data if old is cached. If older season file doesn't exists we get out via an exception.
        if dt.datetime.fromtimestamp(os.path.getmtime(file_name)) > file_date_limit or year != current_season:
            get_new = False
    except FileNotFoundError:
        pass

    date, to_date = gm.get_dates_from_season(year)

    if get_new:
        date = date.replace(day=1)
        futures_tuples = []
        weather_api_native = {}
        with futures.ThreadPoolExecutor(300) as executor:
            while date < to_date:
                if date.month in [7, 8, 9, 10]:
                    date = date.replace(month=date.month+1)
                    date = date.replace(day=1)
                    continue
                url = f'http://h-web03.nve.no/APSapi/TimeSeriesReader.svc/MountainWeather/-/{date.isoformat()}/no/true'
                future = executor.submit(lambda: requests.get(url))
                futures_tuples.append((date, 0, future))
                date += dt.timedelta(days=1)


            while len(futures_tuples):
                date, retries, future = futures_tuples.pop()
                response = future.result()
                if response.status_code != requests.codes.ok:
                    if retries < 5:
                        url = f'http://h-web03.nve.no/APSapi/TimeSeriesReader.svc/MountainWeather/-/{date.isoformat()}/no/true'
                        future = executor.submit(lambda: requests.get(url))
                        futures_tuples.insert(0, (date, retries + 1, future))
                    else:
                        print(f"Failed to fetch weather for {date.isoformat()}, skipping", file=sys.stderr)
                    continue

                json = response.json()
                for obs in json:
                    region = int(float(obs['RegionId']))
                    if region not in REGIONS:
                        continue
                    if obs['Attribute'] not in weather_api_native:
                        weather_api_native[obs['Attribute']] = {}
                    region = int(float(obs['RegionId']))
                    try:
                        weather_api_native[obs['Attribute']][(date.isoformat(), region)] = float(obs['Value'])
                    except ValueError:
                        weather_api_native[obs['Attribute']][(date.isoformat(), region)] = obs['Value']

        with open(file_name, 'wb') as handle:
                pickle.dump(weather_api_native, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(file_name, 'rb') as handle:
                weather_api_native = pickle.load(handle)
        except:
            os.remove(file_name)
            return _get_weather_obs(year, max_file_age)

    weather_api = {}
    for key, (orig_key, mapper) in WEATHER_API.items():
        weather_api[key] = {}
        if orig_key in weather_api_native:
            for date_region, value in weather_api_native[orig_key].items():
                weather_api[key][date_region] = mapper(value)

    weather_varsom = {}
    for forecast in aw:
        for key, (orig_key, mapper) in WEATHER_VARSOM.items():
            if key not in weather_varsom:
                weather_varsom[key] = {}
            date_region = (forecast.date_valid.isoformat(), forecast.region_id)
            weather_varsom[key][date_region] = mapper(getattr(forecast.mountain_weather, orig_key))

    return _merge(weather_varsom, weather_api)


def _get_regobs_obs(regions, year, requested_types, max_file_age=23):
    observations = {}

    if len(requested_types) == 0:
        return observations

    file_name = f'{se.local_storage}regobs_v{CSV_VERSION}_{year}.pickle'
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

    results = []
    if get_new:
        future_tuples = []

        first = requests.post(url=url, json=query).json()
        results = results + first["Results"]
        total_matches = first['TotalMatches']
        searched = number_of_records

        with futures.ThreadPoolExecutor(140) as executor:
            while searched < total_matches:
                query["Offset"] += number_of_records
                query_copy = query.copy()
                future = executor.submit(lambda: requests.post(url=url, json=query_copy))
                future_tuples.append((query_copy, query["Offset"], 0, future))
                searched += number_of_records

            while len(future_tuples):
                query, offset, retries, future = future_tuples.pop()
                try:
                    raw_obses = future.result().json()
                except:
                    if retries < 5:
                        future = executor.submit(lambda: requests.post(url=url, json=query))
                        future_tuples.insert(0, (query, query["Offset"], retries + 1, future))
                    else:
                        print(f"Failed to fetch regobs, offset {offset}, skipping", file=sys.stderr)
                    continue
                results = results + raw_obses["Results"]

        for raw_obs in results:
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
                key = (date.isoformat(), raw_obs["ForecastRegionTid"])
                if key not in observations:
                    observations[key] = {}
                if obs_type not in observations[key]:
                    observations[key][obs_type] = []
                observations[key][obs_type].append(obs)

        # We want the most competent observations first
        for date_region in observations.values():
            for reg_type in date_region.values():
                reg_type.sort(key=lambda x: x['competence'], reverse=True)

        df_dict = {}
        for key, observation in observations.items():
            # Use 5 most competent observations, and list both categories as well as scalars
            for obs_idx in range(0, 5):
                # One type of observation (test, danger signs etc.) at a time
                for regobs_type in requested_types:
                    obses = observation[regobs_type] if regobs_type in observation else []
                    # Go through each requested class attribute from the specified observation type
                    for attr, cat in REGOBS_CLASSES[regobs_type].items():
                        # We handle categories using 1-hot, so we step through each category
                        for cat_name in cat.values():
                            attr_name = f"regobs_{REG_ENG[regobs_type]}_{_camel_to_snake(attr)}_{cat_name}_{obs_idx}"
                            if attr_name not in df_dict:
                                df_dict[attr_name] = {}
                            df_dict[attr_name][key] = obses[obs_idx][cat_name] if len(obses) > obs_idx else 0
                    # Go through all requested scalars
                    for attr, (regobs_attr, conv) in REGOBS_SCALARS[regobs_type].items():
                        attr_name = f"regobs_{REG_ENG[regobs_type]}_{_camel_to_snake(attr)}_{obs_idx}"
                        if attr_name not in df_dict:
                            df_dict[attr_name] = {}
                        try:
                            df_dict[attr_name][key] = conv(obses[obs_idx][regobs_attr]) if len(obses) > obs_idx else 0
                        except TypeError:
                            df_dict[attr_name][key] = 0

            if "accuracy" not in df_dict:
                df_dict["accuracy"] = {}
            df_dict['accuracy'][key] = sum(map(
                lambda x: {0: 0, 1: 1, 2: -1, 3: -1}[x['ForecastCorrectTID']],
                observation['Skredfarevurdering']
            )) if 'Skredfarevurdering' in observation else 0

        with open(file_name, 'wb') as handle:
            pickle.dump(df_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(file_name, 'rb') as handle:
                df_dict = pickle.load(handle)
        except:
            os.remove(file_name)
            return _get_regobs_obs(regions, year, requested_types, max_file_age)

    return df_dict


_camel_re_1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re_2 = re.compile('([a-z0-9])([A-Z])')


def _round_hours(hour):
    if hour >= 21 or hour < 3:
        return 0
    if hour < 9:
        return 6
    if hour < 15:
        return 12
    return 18

def _camel_to_snake(name):
    name = _camel_re_1.sub(r'\1_\2', name)
    return _camel_re_2.sub(r'\1_\2', name).lower()


def _merge(b, a):
    for key in a:
        if key in b and isinstance(a[key], dict) and isinstance(b[key], dict):
            b[key] = _merge(b[key], a[key])
        else:
            b[key] = a[key]
    return b


class Error(Exception):
    pass


class RegObsRegTypeError(Error):
    pass


class CsvMissingError(Error):
    pass
