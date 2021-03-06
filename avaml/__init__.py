import os
import sys

old_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/../varsomdata")
import setenvironment
import varsomdata
os.chdir(old_dir)

_NONE = ""

CSV_VERSION = "32"

REGIONS = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018,
           3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036,
           3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046]

REGION_NEIGH = {
    3001: [3002, 3003, 3004],
    3002: [3001, 3003, 3004],
    3003: [3001, 3002, 3004],
    3004: [3001, 3002, 3003],
    3005: [3006, 3008],
    3006: [3005, 3007, 3008],
    3007: [3006, 3008, 3009],
    3008: [3006, 3007, 3009],
    3009: [3007, 3008, 3010, 3011, 3013],
    3010: [3007, 3008, 3009, 3011, 3013],
    3011: [3007, 3008, 3009, 3010, 3012, 3013],
    3012: [3011, 3013, 3014, 3015],
    3013: [3009, 3010, 3011, 3012, 3015],
    3014: [3012, 3015],
    3015: [3012, 3013, 3014, 3016],
    3016: [3015, 3017],
    3017: [3016, 3018],
    3018: [3017, 3019],
    3019: [3018, 3020],
    3020: [3019, 3021, 3022, 3043],
    3021: [3020, 3022, 3023],
    3022: [3020, 3021, 3023, 3025, 3043],
    3023: [3021, 3022, 3024, 3025],
    3024: [3023, 3025, 3026, 3027, 3028],
    3025: [3022, 3023, 3024, 3028, 3042, 3043],
    3026: [3024, 3027, 3030],
    3027: [3024, 3026, 3028, 3029, 3030, 3031],
    3028: [3024, 3025, 3027, 3029, 3032, 3042],
    3029: [3027, 3028, 3031, 3032],
    3030: [3026, 3027, 3031, 3033],
    3031: [3027, 3029, 3030, 3032, 3033, 3034],
    3032: [3028, 3029, 3031, 3034, 3041, 3042],
    3033: [3030, 3031, 3034, 3036, 3037],
    3034: [3031, 3032, 3033, 3035, 3036, 3037, 3041],
    3035: [3034, 3037, 3038, 3039, 3041],
    3036: [3033, 3034, 3037, 3038],
    3037: [3033, 3034, 3035, 3036, 3038],
    3038: [3035, 3036, 3037, 3039],
    3039: [3035, 3038, 3040, 3041],
    3040: [3039, 3041, 3046],
    3041: [3032, 3034, 3035, 3039, 3040, 3042, 3044, 3045, 3046],
    3042: [3025, 3028, 3032, 3041, 3043, 3044, 3045],
    3043: [3020, 3022, 3025, 3042, 3044],
    3044: [3041, 3042, 3043, 3045, 3046],
    3045: [3041, 3042, 3044],
    3046: [3040, 3041, 3044],
}

REGION_ELEV = {
    3001: (300.0, 600.0),# Copied from 3001
    3002: (300.0, 600.0),# Copied from 3001
    3003: (300.0, 600.0),
    3004: (300.0, 600.0),# Copied from 3001
    3005: (200.0, 500.0),# Copied from 3006
    3006: (200.0, 500.0),
    3007: (300.0, 600.0),
    3008: (400.0, 800.0),# Copied from 3013
    3009: (400.0, 700.0),
    3010: (400.0, 800.0),
    3011: (400.0, 700.0),
    3012: (400.0, 800.0),
    3013: (400.0, 800.0),
    3014: (400.0, 700.0),
    3015: (500.0, 900.0),
    3016: (500.0, 900.0),
    3017: (500.0, 900.0),
    3018: (500.0, 900.0),# Copied from 3017
    3019: (800.0, 1200.0),# Copied from 3022
    3020: (800.0, 1200.0),# Copied from 3022
    3021: (700.0, 1200.0),# Copied from 3023
    3022: (800.0, 1200.0),
    3023: (700.0, 1200.0),
    3024: (700.0, 1200.0),
    3025: (800.0, 1200.0),# Copied from 3022
    3026: (700.0, 1200.0),# Copied from 3024
    3027: (800.0, 1100.0),
    3028: (1000.0, 1600.0),
    3029: (900.0, 1300.0),
    3030: (700.0, 1200.0),# Copied from 3024
    3031: (800.0, 1100.0),
    3032: (1000.0, 1300.0),
    3033: (800.0, 1100.0),# Copied from 3031
    3034: (800.0, 1200.0),
    3035: (1000.0, 1300.0),
    3036: (800.0, 1200.0),# Copied from 3037
    3037: (800.0, 1200.0),
    3038: (500.0, 800.0),# I just gave up being realistic for these regions
    3039: (500.0, 800.0),# I just gave up being realistic for these regions
    3040: (500.0, 800.0),# I just gave up being realistic for these regions
    3041: (500.0, 800.0),# I just gave up being realistic for these regions
    3042: (500.0, 800.0),# I just gave up being realistic for these regions
    3043: (500.0, 800.0),# I just gave up being realistic for these regions
    3044: (500.0, 800.0),# I just gave up being realistic for these regions
    3045: (500.0, 800.0),# I just gave up being realistic for these regions
    3046: (500.0, 800.0),# I just gave up being realistic for these regions
}

class Error(Exception):
    pass

def merge(b, a):
    """
    Merges two dicts. Precedence is given to the second dict. The first dict will be overwritten.
    """
    for key in a:
        if key in b and isinstance(a[key], dict) and isinstance(b[key], dict):
            b[key] = merge(b[key], a[key])
        elif a[key] is not None:
            b[key] = a[key]
    return b
