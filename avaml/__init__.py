import os
import sys

old_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{os.path.dirname(os.path.abspath(__file__))}/../varsomdata")
import setenvironment
import varsomdata
os.chdir(old_dir)

_NONE = ""

CSV_VERSION = "29"

REGIONS = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018,
           3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036,
           3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046]

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
