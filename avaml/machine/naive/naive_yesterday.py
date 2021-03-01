import numpy as np
import pandas as pd

from avaml.aggregatedata.download import CAUSES, DIRECTIONS
from avaml.machine import BulletinMachine

__author__ = 'arwi'

class NaiveYesterday(BulletinMachine):
    def __init__(self):
        self.fitted = True
        super().__init__()

    def fit(self, labeled_data):
        """Does nothing. Here for compability."""

    def predict(self, labeled_data, force_subprobs=False):
        """Predict data using supplied LabeledData.

        :param labeled_data: LabeledData. Dataset to predict. May have empty LabeledData.label.
        :return:             LabeledData. A copy of data, with LabeledData.pred filled in.
        """
        labeled_data = labeled_data.denormalize()
        label = labeled_data.data.reorder_levels([1, 0], axis=1)["1"]

        main_class = ["danger_level", "emergency_warning", "problem_1", "problem_2", "problem_3", "problem_amount"]
        subprobs = ["drift-slab", "glide", "new-loose", "new-slab", "pwl-slab", "wet-loose", "wet-slab", ]
        subprob_class = ["cause", "dist", "dsize", "lev_fill", "prob", "trig"]
        subprob_multi = ["aspect"]
        subprob_real = ["lev_min", "lev_max"]
        columns = pd.MultiIndex.from_product([["CLASS"], [""], main_class])
        columns = columns.append(pd.MultiIndex.from_product([["CLASS"], subprobs, subprob_class]))
        columns = columns.append(pd.MultiIndex.from_product([["MULTI"], subprobs, subprob_multi]))
        columns = columns.append(pd.MultiIndex.from_product([["REAL"], subprobs, subprob_real]))
        out = pd.DataFrame(index=labeled_data.data.index, columns=columns)

        out["CLASS", ""] = ""
        out["CLASS", "", "danger_level"] = label["danger_level"].apply(int)
        out["CLASS", "", "problem_amount"] = label["problem_amount"].apply(int)
        out.loc[label["emergency_warning"] == 0, ("CLASS", "", "emergency_warning")] = "Naturlig utløste skred"
        out.loc[label["emergency_warning"] != 0, ("CLASS", "", "emergency_warning")] = "Ikke gitt"
        label_problems = np.array(subprobs)[(-label[list(map(lambda x: "problem_" + x, subprobs))]).values.argsort()]
        p_1 = label["problem_amount"] > 0
        p_2 = label["problem_amount"] > 1
        p_3 = label["problem_amount"] > 2
        out.loc[p_1, ("CLASS", "", "problem_1")] = label_problems[p_1, 0]
        out.loc[p_2, ("CLASS", "", "problem_2")] = label_problems[p_2, 1]
        out.loc[p_3, ("CLASS", "", "problem_3")] = label_problems[p_3, 2]

        for subprob in subprobs:
            out["CLASS", subprob, "cause"] = np.array(list(CAUSES.values()))[
                label[list(map(lambda x: f"problem_{subprob}_cause_{x}", CAUSES.values()))].values.argmax(axis=1)
            ]
            out["CLASS", subprob, "dsize"] = label[f"problem_{subprob}_dsize"].apply(int)
            out["CLASS", subprob, "prob"] = label[f"problem_{subprob}_prob"].apply(int)
            out["CLASS", subprob, "trig"] = label[f"problem_{subprob}_trig"].apply(int).replace({0: 10, 1: 21, 2: 22})
            out["CLASS", subprob, "dist"] = label[f"problem_{subprob}_dist"].apply(int)
            out["CLASS", subprob, "lev_fill"] = np.array(range(1, 5))[
                label[list(map(lambda x: f"problem_{subprob}_lev_fill_{x}", range(1, 5)))].values.argmax(axis=1)
            ].astype(int)
            out["MULTI", subprob, "aspect"] = label[
                [f"problem_{subprob}_aspect_{dir}" for dir in DIRECTIONS]
            ].apply(lambda x: "".join(x.apply(int).apply(str).to_list()), axis=1)
            out["REAL", subprob, "lev_min"] = label[f"problem_{subprob}_lev_min"]
            out["REAL", subprob, "lev_max"] = label[f"problem_{subprob}_lev_max"]

        ld = labeled_data.copy()
        ld.pred = out
        return ld.valid_pred()

    def dump(self, identifier):
        """Does nothing. Here for compability."""

    @staticmethod
    def load(identifier):
        return NaiveYesterday()
