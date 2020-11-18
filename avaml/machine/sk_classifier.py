import re

import numpy as np
import pandas as pd
import dill

from avaml import _NONE, setenvironment as se
from avaml.aggregatedata import DatasetMissingLabel
from avaml.machine import BulletinMachine, AlreadyFittedError, NotFittedError

__author__ = 'arwi'

DILL_VERSION = '3'

class SKClassifierMachine(BulletinMachine):
    def __init__(
            self,
            ml_prim_creator,
            ml_class_creator,
            ml_multi_creator,
            ml_real_creator,
            sk_prim_class_weight=None,
            sk_class_weight=None,
    ):
        """Facilitates training and prediction of avalanche warnings. Only supports sklearn models.

        :param ml_prim_creator:      fn(in_size: Int, out_size: Int) -> classifier: Used to solve primary problems,
                                     such as "danger_level" and "problem_n".
        :param ml_class_creator:     fn(in_size: Int, out_size: Int) -> classifier: Used to solve secondary problems,
                                     such as "cause" or "dist".
        :param ml_multi_creator:     fn(in_size: Int, out_size: Int) -> classifier: Used to solve multilabel problems,
                                     such as "aspect". Must be k-of-n-hot.
        :param ml_real_creator:      fn(in_size: Int, out_size: Int) -> regressor: Used to solve for real numbers. Must
                                     support multiple outputs.
        :param sk_prim_class_weight: Class weights for "danger_level", "emergency_warning" and "problem_<n>".
                                     Either None, "balanced", "balanced_subsample" or dict of type
                                     {"danger_level": {'4': 2}}.
        :param sk_class_weight:      Class weights for subproblems.
                                     Either None, "balanced", "balanced_subsample" or dict of type
                                     {"cause": {'new-snow': 2}}.
        """
        self.ml_prim_creator = ml_prim_creator
        self.ml_class_creator = ml_class_creator
        self.ml_multi_creator = ml_multi_creator
        self.ml_real_creator = ml_real_creator
        self.sk_prim_class_weight = sk_prim_class_weight
        self.sk_class_weight = sk_class_weight
        self.machines_class = {}
        self.machines_multi = {}
        self.machines_real = {}
        self.labels_class = {}
        self.labels_multi = {}
        self.labels_real = {}
        super().__init__()

    def fit(self, labeled_data):
        """Fits models to the supplied LabeledData.

        :param labeled_data: LabeledData: Dataset that the models should be fit after.
        """
        if labeled_data.label is None:
            raise DatasetMissingLabel()

        if self.fitted:
            raise AlreadyFittedError()
        self.fitted = True

        X = labeled_data.data
        y = labeled_data.label
        dummies = labeled_data.to_dummies()['label']

        self.X_columns = X.columns
        self.y_columns = y.columns
        self.dummies_columns = dummies.columns
        self.dtypes = y.dtypes.to_dict()
        self.row_weight = labeled_data.row_weight

        try:
            prob_cols = y.loc[:, [name.startswith("problem_") for name in y.columns.get_level_values(2)]]["CLASS", ""]
        except KeyError:
            prob_cols = pd.DataFrame(index=y.index)
        for subprob in dummies["CLASS"].columns.get_level_values(0).unique():
            # Special machine for danger level etc.
            if subprob == _NONE:
                idx = [True] * y.shape[0]
                machine_creator = self.ml_prim_creator
                class_weight = self.sk_prim_class_weight
                prepared_weight = prepare_class_weight_(class_weight, dummies["CLASS", subprob])
            else:
                idx = np.any(np.char.equal(prob_cols.values.astype("U"), subprob), axis=1)
                machine_creator = self.ml_class_creator
                class_weight = self.sk_class_weight

                # "0" has special meaning of no-data in these columns.
                weight_dummy = dummies["CLASS", subprob]
                weight_dummy = weight_dummy.loc[:, weight_dummy.columns.get_level_values(1) != "0"]
                prepared_weight = prepare_class_weight_(class_weight, weight_dummy)

            if np.sum(idx):
                self.machines_class[subprob] = machine_creator(
                    X.shape[1:],
                    y["CLASS", subprob].shape[1],
                    class_weight=prepared_weight
                )
                try:
                    self.machines_class[subprob].fit(
                        X[idx],
                        y.loc[idx, pd.IndexSlice["CLASS", subprob]],
                        sample_weight=np.ravel(self.row_weight.loc[idx])
                    )
                except TypeError:
                    self.machines_class[subprob].fit(
                        X[idx],
                        y.loc[idx, pd.IndexSlice["CLASS", subprob]]
                    )

        try:
            for subprob in dummies["MULTI"].columns.get_level_values(0).unique():
                idx = np.any(np.char.equal(prob_cols.values.astype("U"), subprob), axis=1)
                self.machines_multi[subprob] = self.ml_multi_creator(
                    X.shape[1:],
                    y["CLASS", subprob].shape[1]
                )

                if np.sum(idx):
                    try:
                        self.machines_multi[subprob].fit(
                            X[idx],
                            dummies.loc[idx, pd.IndexSlice["MULTI", subprob]],
                            sample_weight=np.ravel(self.row_weight.loc[idx])
                        )
                    except TypeError:
                        self.machines_multi[subprob].fit(
                            X[idx],
                            dummies.loc[idx, pd.IndexSlice["MULTI", subprob]]
                        )
        except KeyError:
            pass

        try:
            for subprob in y["REAL"].columns.get_level_values(0).unique():
                idx = np.any(np.char.equal(prob_cols.values.astype("U"), subprob), axis=1)
                self.machines_real[subprob] = self.ml_real_creator(X.shape[1:], y["REAL", subprob].shape[1])

                if np.sum(idx):
                    try:
                        self.machines_real[subprob].fit(
                            X[idx],
                            y.loc[idx, pd.IndexSlice["REAL", subprob]].astype(np.float),
                            sample_weight=np.ravel(self.row_weight.loc[idx])
                        )
                    except TypeError:
                        self.machines_real[subprob].fit(
                            X[idx],
                            y.loc[idx, pd.IndexSlice["REAL", subprob]].astype(np.float)
                        )
        except KeyError:
            pass

    def predict(self, labeled_data):
        """Predict data using supplied LabeledData.

        :param labeled_data: LabeledData. Dataset to predict. May have empty LabeledData.label.
        :return:             LabeledData. A copy of data, with LabeledData.pred filled in.
        """
        if not self.fitted:
            raise NotFittedError()

        X = labeled_data.data.values
        y = pd.DataFrame(
            index=labeled_data.data.index,
            columns=self.y_columns
        ).fillna(0).astype(self.dtypes)
        y.loc[:, y.dtypes == np.object] = _NONE

        y["CLASS", _NONE] = self.machines_class[""].predict(X)
        problem_cols = []
        for n in range(1, 4):
            if f"problem_{n}" in list(y["CLASS", _NONE].columns):
                problem_cols.append(("CLASS", _NONE, f"problem_{n}"))
        prev_eq = np.zeros((y.shape[0], len(problem_cols)), dtype=bool)
        for n, col in enumerate(problem_cols):
            for mcol in problem_cols[1:n]:
                # If equal to problem_n-1/2, set to _NONE (as we only have digital results).
                prev_eq[:, n] = np.logical_or(
                    prev_eq[:, n],
                    np.equal(y[mcol], y[col])
                )
                # Set to None if problem_n-1/2 was None.
                prev_eq[:, n] = np.logical_or(
                    prev_eq[:, n],
                    y[mcol] == _NONE
                )
            y.loc[prev_eq[:, n], col] = _NONE

        # Calculate relevant subproblems
        for subprob, machine in self.machines_class.items():
            if subprob == _NONE:
                continue
            rows = np.any(np.char.equal(y[problem_cols].values.astype("U"), subprob), axis=1)
            if np.sum(rows):
                pred = machine.predict(X[np.ix_(rows)])
                classes = self.dummies_columns.to_frame().loc[pd.IndexSlice["CLASS", subprob]].iloc[:, 2].unique()
                for n, attr in enumerate(classes):
                    y["CLASS", subprob, attr].values[np.ix_(rows)] = pred[:, n]
        for subprob, machine in self.machines_multi.items():
            rows = np.any(np.char.equal(y[problem_cols].values.astype("U"), subprob), axis=1)
            if np.sum(rows):
                pred = machine.predict(X[np.ix_(rows)])
                multis = self.dummies_columns.to_frame().loc[pd.IndexSlice["MULTI", subprob]].iloc[:, 2].unique()
                for attr in multis:
                    labels = pred.astype(np.int).astype("U")
                    y["MULTI", subprob, attr].values[np.ix_(rows)] = [''.join(row) for row in labels]
        for subprob, machine in self.machines_real.items():
            rows = np.any(np.char.equal(y[problem_cols].values.astype("U"), subprob), axis=1)
            if np.sum(rows):
                pred = machine.predict(X[np.ix_(rows)])
                reals = self.dummies_columns.to_frame().loc[pd.IndexSlice["REAL", subprob]].iloc[:, 2].unique()
                for n, attr in enumerate(reals):
                    y["REAL", subprob, attr].values[np.ix_(rows)] = pred[:, n]

        df = labeled_data.copy()
        df.pred = y
        return df


    def feature_importances(self):
        """Used to get all feature importances of internal classifiers.
        Supplied models must support model.feature_importances_, otherwise they are ignored.

        :return: DataFrame. Feature importances of internal classifiers.
        """
        importances = {}
        for attr, machine in self.machines_class.items():
            try:
                importances[('CLASS', attr)] = machine.feature_importances_
            except AttributeError:
                pass
        for attr, machine in self.machines_multi.items():
            try:
                importances[('MULTI', attr)] = machine.feature_importances_
            except AttributeError:
                pass
        for attr, machine in self.machines_real.items():
            try:
                importances[('REAL', attr)] = machine.feature_importances_
            except AttributeError:
                pass
        df = pd.DataFrame(importances, index=self.X_columns)
        df.index.set_names(["feature_name", "day"], inplace=True)
        return df


    def dump(self, identifier):
        file_name = f'{se.local_storage}model_skclass_v{DILL_VERSION}_{identifier}.dill'

        with open(file_name, 'wb') as handle:
            dill.dump(self, handle)

    @staticmethod
    def load(identifier):
        file_name = f'{se.local_storage}model_skclass_v{DILL_VERSION}_{identifier}.dill'
        with open(file_name, 'rb') as handle:
            bm = dill.load(handle)
        return bm


def prepare_class_weight_(class_weight, dummies):
    if class_weight is None:
        return None
    if class_weight == "balanced" or class_weight == "balanced_subsample":
        return class_weight

    prepared_weight = []
    for attribute in dummies.columns.get_level_values(0).unique():
        attr_weights = {}
        for label in dummies[attribute].columns:
            if attribute in class_weight and label in class_weight[attribute]:
                weight = class_weight[attribute][label]
                attr_weights[label] = weight
            else:
                attr_weights[label] = 1
        prepared_weight.append(attr_weights)
    return prepared_weight
