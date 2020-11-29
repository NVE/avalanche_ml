import os
import pickle
import re
import sys
import datetime as dt
import requests
from concurrent import futures

from avaml import _NONE, CSV_VERSION, REGIONS, merge, Error, setenvironment as se, varsomdata
from varsomdata import getforecastapi as gf
from varsomdata import getvarsompickles as gvp
from varsomdata import getmisc as gm

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
    # Mock. This value doesn't exist in APS, so we let varsom through by specifying None.
    "wind_change_speed": ("WindClassification", lambda x: None),
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
    # Mocks. These value doesn't exist in APS, so we let varsom through by specifying None.
    "wind_chg_dir_N": ("WindDirection", lambda x: None),
    "wind_chg_dir_NE": ("WindDirection", lambda x: None),
    "wind_chg_dir_E": ("WindDirection", lambda x: None),
    "wind_chg_dir_SE": ("WindDirection", lambda x: None),
    "wind_chg_dir_S": ("WindDirection", lambda x: None),
    "wind_chg_dir_SW": ("WindDirection", lambda x: None),
    "wind_chg_dir_W": ("WindDirection", lambda x: None),
    "wind_chg_dir_NW": ("WindDirection", lambda x: None),
    "wind_chg_start_0": ("WindDirection", lambda x: None),
    "wind_chg_start_6": ("WindDirection", lambda x: None),
    "wind_chg_start_12": ("WindDirection", lambda x: None),
    "wind_chg_start_18": ("WindDirection", lambda x: None),
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


def _get_raw_varsom(year, date, days, max_file_age=23):
    if date:
        season = gm.get_season_from_date(date)
        regions = gm.get_forecast_regions(year=season, get_b_regions=True)
        aw = []
        from_date = date - dt.timedelta(days=days + 1)
        to_date = date
        single_warning = gf.get_avalanche_warnings(regions, from_date, to_date)
        for sw in single_warning:
            if sw.danger_level > 0:
                aw.append(sw)
    else:
        aw = gvp.get_all_forecasts(year=year, max_file_age=max_file_age)
    return aw


def _get_varsom_obs(year, date=None, days=None, max_file_age=23):
    """
    Download data from Varsom
    :param year: String representation of season. None if a specific date should be fetched.
    :param date: datetime.date. None if a whole season should be fetched.
    :param days: How many days to fetch before date. This will be max for .label()'s days parameter.
    """
    aw = _get_raw_varsom(year, date, days, max_file_age=max_file_age)
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

def _get_weather_obs(year, date=None, days=None, max_file_age=23):
    """
    Download data from the weather API:s
    :param year: String representation of season. None if a specific date should be fetched.
    :param max_file_age: Time to live for cache in hours.
    :param date: datetime.date. None if a whole season should be fetched.
    """
    aw = _get_raw_varsom(year, date, days, max_file_age=max_file_age)
    file_name = f'{se.local_storage}weather_v{CSV_VERSION}_{year}.pickle'
    file_date_limit = dt.datetime.now() - dt.timedelta(hours=max_file_age)
    current_season = gm.get_season_from_date(dt.date.today() - dt.timedelta(30))
    get_new = True

    if date:
        from_date = date - dt.timedelta(days=days)
        to_date = date + dt.timedelta(days=1)
        get_new = True
    else:
        from_date, to_date = gm.get_dates_from_season(year)
        to_date = to_date + dt.timedelta(days=1)

        try:
            # Don't fetch new data if old is cached. If older season file doesn't exists we get out via an exception.
            if dt.datetime.fromtimestamp(os.path.getmtime(file_name)) > file_date_limit or year != current_season:
                get_new = False
        except FileNotFoundError:
            pass

    if get_new:
        futures_tuples = []
        weather_api_native = {}
        with futures.ThreadPoolExecutor(300) as executor:
            while from_date < to_date:
                if from_date.month in [7, 8, 9, 10]:
                    from_date = from_date.replace(day=1, month=from_date.month + 1)
                    continue
                url = 'http://h-web03.nve.no/APSapi/TimeSeriesReader.svc/MountainWeather/-/{0}/no/true'.format(
                    from_date.isoformat()
                )
                future = executor.submit(lambda: requests.get(url))
                futures_tuples.append((from_date, 0, future))
                from_date += dt.timedelta(days=1)


            while len(futures_tuples):
                from_date, retries, future = futures_tuples.pop()
                response = future.result()
                if response.status_code != requests.codes.ok:
                    if retries < 5:
                        url = 'http://h-web03.nve.no/APSapi/TimeSeriesReader.svc/MountainWeather/-/{0}/no/true'.format(
                            from_date.isoformat()
                        )
                        future = executor.submit(lambda: requests.get(url))
                        futures_tuples.insert(0, (from_date, retries + 1, future))
                    else:
                        print(f"Failed to fetch weather for {from_date.isoformat()}, skipping", file=sys.stderr)
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
                        weather_api_native[obs['Attribute']][(from_date.isoformat(), region)] = float(obs['Value'])
                    except ValueError:
                        weather_api_native[obs['Attribute']][(from_date.isoformat(), region)] = obs['Value']

        if not date:
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

    return merge(weather_varsom, weather_api)


def _get_regobs_obs(year, requested_types, date=None, days=None, max_file_age=23):
    regions = gm.get_forecast_regions(year=year, get_b_regions=True)
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
        if date:
            get_new = True
    except FileNotFoundError:
        pass

    if date:
        from_date = date - dt.timedelta(days=days)
        to_date = date
    else:
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

        if not date:
            with open(file_name, 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            with open(file_name, 'rb') as handle:
               results = pickle.load(handle)
        except:
            os.remove(file_name)
            return _get_regobs_obs(regions, year, requested_types, max_file_age)

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

    return df_dict


def _round_hours(hour):
    if hour >= 21 or hour < 3:
        return 0
    if hour < 9:
        return 6
    if hour < 15:
        return 12
    return 18


_camel_re_1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re_2 = re.compile('([a-z0-9])([A-Z])')

def _camel_to_snake(name):
    name = _camel_re_1.sub(r'\1_\2', name)
    return _camel_re_2.sub(r'\1_\2', name).lower()


class RegObsRegTypeError(Error):
    pass


