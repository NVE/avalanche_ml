from functools import reduce
import re

import numpy as np
import pandas as pd

from avaml import _NONE
from avaml.aggregatedata import DatasetMissingLabel
from avaml.score.overlap import calc_overlap

__author__ = 'arwi'

VECTOR_WETNESS_LOOSE = {
    _NONE: (0, 0),
    "new-loose": (0, 1),
    "wet-loose": (1, 1),
    "new-slab": (0, 0.4),
    "drift-slab": (0, 0.2),
    "pwl-slab": (0, 0),
    "wet-slab": (1, 0),
    "glide": (0.8, 0),
}

VECTOR_FREQ = {
    "dsize": {
        _NONE: 0,
        '0': 0,
        '1': 0.2,
        '2': 0.4,
        '3': 0.6,
        '4': 0.8,
        '5': 1,
    },
    "dist": {
        _NONE: 0,
        '0': 0,
        '1': 0.25,
        '2': 0.5,
        '3': 0.75,
        '4': 1,
    },
    "trig": {
        _NONE: 0,
        '0': 0,
        '10': 1 / 3,
        '21': 2 / 3,
        '22': 1,
    },
    "prob": {
        _NONE: 0,
        '0': 0,
        '2': 1 / 3,
        '3': 2 / 3,
        '5': 1,
    },
}

class Score:
    def __init__(self, labeled_data):
        def to_vec(df):
            level_2 = ["wet", "loose", "freq", "lev_max", "lev_min", "lev_fill", "aspect"]
            columns = pd.MultiIndex.from_product([["global"], ["danger_level", "emergency_warning"]]).append(
                pd.MultiIndex.from_product([[f"problem_{n}" for n in range(1, 4)], level_2])
            )
            vectors = pd.DataFrame(index=df.index, columns=columns)
            vectors[("global", "danger_level")] = df[("CLASS", _NONE, "danger_level")].astype(np.int) / 5
            vectors[("global", "emergency_warning")] = (
                    df[("CLASS", _NONE, "emergency_warning")] == "Naturlig utløste skred"
            ).astype(np.int)
            for idx, row in df.iterrows():
                for prob_n in [f"problem_{n}" for n in range(1, 4)]:
                    problem = row["CLASS", _NONE, prob_n]
                    if problem == _NONE:
                        vectors.loc[idx, prob_n] = [0, 0, 0, 0, 0, 2, "00000000"]
                    else:
                        p_class = row["CLASS", problem]
                        p_real = row["REAL", problem]
                        wet = VECTOR_WETNESS_LOOSE[problem][0]
                        loose = VECTOR_WETNESS_LOOSE[problem][1]
                        freq = reduce(lambda x, y: x * VECTOR_FREQ[y][p_class[y]], VECTOR_FREQ.keys(), 1)
                        lev_max = float(p_real["lev_max"]) if p_real["lev_max"] else 0.0
                        lev_min = float(p_real["lev_min"]) if p_real["lev_min"] else 0.0
                        lev_fill = int(p_class["lev_fill"]) if p_class["lev_fill"] else 0
                        aspect = row["MULTI", problem, "aspect"]
                        vectors.loc[idx, prob_n] = [wet, loose, freq, lev_max, lev_min, lev_fill, aspect]
            return vectors

        if labeled_data.label is None or labeled_data.pred is None:
            raise DatasetMissingLabel()
        self.label_vectors = to_vec(labeled_data.label)
        self.pred_vectors = to_vec(labeled_data.pred)

    def calc(self):
        diff_cols = [not re.match(r"^(lev_)|(aspect)", col) for col in self.label_vectors.columns.get_level_values(1)]
        diff = self.pred_vectors.loc[:, diff_cols] - self.label_vectors.loc[:, diff_cols]
        p_score_cols = pd.MultiIndex.from_tuples([(_NONE, "problem_score")]).append(
            pd.MultiIndex.from_product([[f"problem_{n}" for n in range(1, 4)], ["spatial_diff"]])
        )
        p_score = pd.DataFrame(index=diff.index, columns=p_score_cols)
        for idx, series in self.label_vectors.iterrows():
            problem_score, spatial_diffs = dist(series, self.pred_vectors.loc[idx])
            p_score.loc[idx] = np.array([problem_score] + spatial_diffs)
        weights = np.array([1, 1, 1])
        maxdist = np.power(weights, 2).sum()
        score = np.power(
            pd.concat([diff.iloc[:, :2], p_score[[("", "problem_score")]]], axis=1).astype(np.float) * weights,
            2
        ).sum(axis=1)
        score = pd.DataFrame(
            pd.concat([score / maxdist, np.sqrt(score) / np.sqrt(maxdist)], axis=1).values,
            index=diff.index,
            columns=pd.MultiIndex.from_tuples([(_NONE, "score"), (_NONE, "distance")])
        )
        return pd.concat([score, p_score, diff], axis=1).sort_index(axis=1)

def dist(score1, score2):
    def distvec(vec1, vec2):
        spatial_diff = 1 - calc_overlap(vec1, vec2)

        # Distance to the null forecast is the distance to the plane freq = 0.
        if not vec1["freq"]:
            return vec2["freq"], spatial_diff
        if not vec2["freq"]:
            return vec1["freq"], spatial_diff

        diff = np.append(vec1.iloc[:3] - vec2.iloc[:3], [spatial_diff])
        # With two valid forecasts, compute a weighted euclidean norm.
        norm = np.linalg.norm((diff) * np.array([1, 1, 1, 0.5]))
        return norm, spatial_diff

    def mindist(vec1, vecs):
        mindist = None
        for vec2 in vecs:
            dist, _ = distvec(vec1, vec2)
            if dist > 0 and (mindist is None or dist < mindist):
                mindist = dist
        return mindist if mindist is not None else 0

    maxdist = np.linalg.norm(np.array([1, 1, 1, 0.5]))
    distance = 0
    prob_cols = [f"problem_{n}" for n in range(1, 4)]
    weights = [3, 2, 1]
    spatial_diffs = []
    for prob_n, weight in zip(prob_cols, weights):
        norm, spatial_diff = distvec(score1[prob_n], score2[prob_n])
        distance += norm * weight
        dist_1 = mindist(score1[prob_n], [score2[prob_m] for prob_m in prob_cols])
        dist_2 = mindist(score2[prob_n], [score1[prob_m] for prob_m in prob_cols])
        distance += (dist_1 + dist_2) * weight / 2
        spatial_diffs.append(spatial_diff)

    return distance / (2 * maxdist * sum(weights)), spatial_diffs
