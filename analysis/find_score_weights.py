"""
Find what weights to use in the Score class to create a balanced metric.
"""
import math

from avaml.aggregatedata.__init__ import LabeledData, CsvMissingError, ForecastDataset
from avaml.score import Score, calc_overlap, dist
import pandas as pd
import numpy as np

# Fetch data
days = 0
regobs_types = []
varsom = False
seasons = ["2017-18", "2018-19", "2019-20"]
try:
    print("Reading csv")
    labeled_data = LabeledData.from_csv(seasons=seasons, days=days, regobs_types=regobs_types, with_varsom=varsom)
except CsvMissingError:
    print("Csv missing. Fetching online data. (This takes a long time.)")
    labeled_data = ForecastDataset(seasons=seasons, regobs_types=regobs_types).label(days=days, with_varsom=varsom)
    labeled_data.to_csv()

print("Calculating scores")
# Score need something at .pred, otherwise it won't run.
labeled_data.pred = labeled_data.label
score = Score(labeled_data)
vectors = score.label_vectors
concatenated_score = pd.concat([vectors["problem_1"], vectors["problem_2"], vectors["problem_3"]])
non_empty = concatenated_score["freq"] != 0
concatenated_problems = concatenated_score.loc[non_empty]

def calc_problem(problems):
    mean_lev_max = np.mean(problems["lev_max"])
    mode_lev_fill = problems["lev_fill"].mode().iloc[0]
    if int(mode_lev_fill) != 1:
        raise Exception("lev_fill != 1")
    mode_aspect = "".join(
        pd.DataFrame(
            problems["aspect"].apply(lambda x: x.zfill(8)).apply(list).to_list()
        ).mode().iloc[0].to_list()
    )
    mean = pd.Series({
        "wet": np.mean(problems["wet"]),
        "loose": np.mean(problems["loose"]),
        "freq": np.mean(problems["freq"]),
        "lev_max": mean_lev_max,
        "lev_min": 0,
        "lev_fill": 1,
        "aspect": mode_aspect
    })
    spatial_diff = problems.apply(lambda x: calc_overlap(x, mean), axis=1)

    std_wet = np.std(problems["wet"])
    std_loose = np.std(problems["loose"])
    std_freq = np.std(problems["freq"])
    std_spatial_diff = np.std(spatial_diff)

    weights = 1 / np.array([std_wet, std_loose, std_freq, std_spatial_diff])
    return mean, weights / np.max(weights)

_, problem_weights = calc_problem(concatenated_problems)
print("Problem weights", problem_weights)

problem_1, _ = calc_problem(vectors["problem_1"].loc[vectors[("problem_1", "freq")] != 0])
problem_2, _ = calc_problem(vectors["problem_2"].loc[vectors[("problem_2", "freq")] != 0])
problem_3, _ = calc_problem(vectors["problem_2"].loc[vectors[("problem_2", "freq")] != 0])
empty = pd.Series({
    "wet": 0,
    "loose": 0,
    "freq": 0,
    "lev_max": 0,
    "lev_min": 0,
    "lev_fill": 2,
    "aspect": "00000000"
})

p_0_percentage = (vectors[("problem_1", "freq")] == 0).sum() / vectors.shape[0]
p_3_percentage = (vectors[("problem_3", "freq")] != 0).sum() / vectors.shape[0]
p_2_percentage = (vectors[("problem_2", "freq")] != 0).sum() / vectors.shape[0] - p_3_percentage
p_1_percentage = 1 - p_0_percentage - p_2_percentage
break_0 = math.floor(vectors.shape[0] * p_0_percentage) + 1
break_1 = math.floor(vectors.shape[0] * p_1_percentage) + 1
break_2 = math.floor(vectors.shape[0] * p_2_percentage) + 1
print("Problem 0, ", p_0_percentage, "%")
print("Problem 1, ", p_1_percentage, "%")
print("Problem 2, ", p_2_percentage, "%")
print("Problem 3, ", p_3_percentage, "%")

n0_problems = pd.Series(pd.concat([pd.Series([0, 0]), empty, empty, empty]).values, index=vectors.columns)
n1_problems = pd.Series(pd.concat([pd.Series([0, 0]), problem_1, empty, empty]).values, index=vectors.columns)
n2_problems = pd.Series(pd.concat([pd.Series([0, 0]), problem_1, problem_2, empty]).values, index=vectors.columns)
n3_problems = pd.Series(pd.concat([pd.Series([0, 0]), problem_1, problem_2, problem_3]).values, index=vectors.columns)
problem_scores = []
for idx, (_, series) in enumerate(vectors.sample(frac=1).iterrows()):
    if idx < break_0:
        problem_score, _ = dist(series, n0_problems)
    if idx < break_1:
        problem_score, _ = dist(series, n1_problems)
    if idx < break_2:
        problem_score, _ = dist(series, n2_problems)
    else:
        problem_score, _ = dist(series, n3_problems)
    problem_scores.append(problem_score)

std_danger_level = np.std(vectors[("global", "danger_level")])
std_emergency_warning = np.std(vectors[("global", "emergency_warning")])
std_problem_score = np.std(np.array(problem_scores))
weights = 1 / np.array([std_danger_level, std_emergency_warning, std_problem_score])
weights = weights / np.max(weights)

print("Weights, ", weights)
