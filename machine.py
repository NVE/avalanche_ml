from aggregatedata import PROBLEMS, _NONE
import os
import sys
import numpy as np
import pandas
import dill
import tensorflow.keras.models as keras

old_dir = os.getcwd()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "./varsomdata")
import setenvironment as se
os.chdir(old_dir)

__author__ = 'arwi'

DILL_VERSION = '2'

class BulletinMachine:
    def __init__(
            self,
            ml_prim_creator,
            ml_class_creator,
            ml_multi_creator,
            ml_real_creator,
            sk_prim_class_weight=None,
            sk_class_weight=None,
    ):
        """Facilitates training and prediction of avalanche warnings.

        :param ml_prim_creator:      fn(in_size: Int, out_size: Int) -> classifier: Used to solve primary problems,
                                     such as "danger_level" and "problem_n". Preferably softmax output.
        :param ml_class_creator:     fn(in_size: Int, out_size: Int) -> classifier: Used to solve secondary problems,
                                     such as "cause" or "dist". Preferably softmax output.
        :param ml_multi_creator:     fn(in_size: Int, out_size: Int) -> classifier: Used to solve multilabel problems,
                                     such as "aspect". Must be k-of-n-hot.
        :param ml_real_creator:      fn(in_size: Int, out_size: Int) -> regressor: Used to solve for real numbers. Must
                                     support multiple outputs.
        :param sk_prim_class_weight: Class weights for "danger_level", "emergency_warning" and "problem_<n>".
                                     Either None, "balanced", "balanced_subsample" or dict of type
                                     {"danger_level": {'4': {0: 2, 1: 2}}}. Only works for sklearn models.
        :param sk_class_weight:      Class weights for subproblems.
                                     Either None, "balanced", "balanced_subsample" or dict of type
                                     {"cause": {'new-snow': {0: 2, 1: 2}}}. Only works for sklearn models.
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
        self.y_columns = None
        self.X_columns = None
        self.dtypes = None
        self.row_weight = None
        self.fitted = False
        self.is_timeseries = {_NONE: False, "CLASS": False, "MULTI": False, "REAL": False}

    def fit(self, labeled_data, epochs, verbose=0):
        """Fits models to the supplied LabeledData.

        :param labeled_data: LabeledData: Dataset that the models should be fit after.
        :param epochs:       Int. Number of epochs to train. Ignored if the supplied model doesn't
                             support the parameter.
        :param verbose:      Int. Verbosity of the models. Ignored if not supported of the supplied
                             models.
        """
        if self.fitted:
            raise AlreadyFittedError()
        self.fitted = True
        labeled_data = labeled_data.normalize()
        y = labeled_data.label
        self.y_columns = y.columns
        self.dtypes = y.dtypes.to_dict()
        self.dummies = labeled_data.to_dummies()['label']
        X = labeled_data.data
        X_timeseries = labeled_data.to_timeseries()[0]
        self.X_columns = X.columns
        self.row_weight = labeled_data.row_weight

        prob_cols = y.loc[:, [name.startswith("problem_") for name in y.columns.get_level_values(2)]]["CLASS", ""]
        for subprob, dummy in self.dummies["CLASS"].items():
            # Special machine for danger level etc.
            if subprob == _NONE:
                idx = [True] * dummy.shape[0]
                machine_creator = self.ml_prim_creator
                class_weight = self.sk_prim_class_weight
            else:
                idx = np.any(np.char.equal(prob_cols.values.astype("U"), subprob), axis=1)
                machine_creator = self.ml_class_creator
                class_weight = self.sk_class_weight

            # See if we're supposed to use dataset as is (2D) or convert to 3D
            try:
                self.machines_class[subprob] = machine_creator(X.shape[1:], len(dummy.columns))
                X_class = X
            except ValueError:
                self.is_timeseries["CLASS" if subprob else _NONE] = True
                self.machines_class[subprob] = machine_creator(X_timeseries.shape[1:], len(dummy.columns))
                X_class = X_timeseries

            if np.sum(idx):
                if "sklearn" in str(type(self.machines_class[subprob])):
                    prepared_weight = prepare_class_weight_(class_weight, dummy)
                    self.machines_class[subprob] = machine_creator(
                        X_class.shape[1:],
                        len(dummy.columns),
                        class_weight=prepared_weight
                    )
                    try:
                        self.machines_class[subprob].fit(
                            X_class[idx],
                            dummy.loc[idx],
                            sample_weight=np.ravel(self.row_weight.loc[idx])
                        )
                    except TypeError:
                        self.machines_class[subprob].fit(X_class[idx], dummy.loc[idx])
                else:
                    self.machines_class[subprob].fit(X_class[idx], dummy.loc[idx], epochs=epochs, verbose=verbose)

        for subprob, dummy in self.dummies["MULTI"].items():
            idx = np.any(np.char.equal(prob_cols.values.astype("U"), subprob), axis=1)

            try:
                self.machines_multi[subprob] = self.ml_multi_creator(X.shape[1:], len(dummy.columns))
                X_multi = X
            except ValueError:
                self.is_timeseries["MULTI"] = True
                self.machines_multi[subprob] = self.ml_multi_creator(X_timeseries.shape[1:], len(dummy.columns))
                X_multi = X_timeseries
            if np.sum(idx):
                if "sklearn" in str(type(self.machines_multi[subprob])):
                    try:
                        self.machines_multi[subprob].fit(
                            X_multi[idx],
                            dummy.loc[idx],
                            sample_weight=np.ravel(self.row_weight.loc[idx])
                        )
                    except TypeError:
                        self.machines_multi[subprob].fit(X_multi[idx], dummy.loc[idx])
                else:
                    self.machines_multi[subprob].fit(X_multi[idx], dummy.loc[idx], epochs=epochs, verbose=verbose)

        for subprob, values in self.dummies["REAL"].items():
            idx = np.any(np.char.equal(prob_cols.values.astype("U"), subprob), axis=1)

            try:
                self.machines_real[subprob] = self.ml_real_creator(X.shape[1:], len(dummy.columns))
                X_real = X
            except ValueError:
                self.is_timeseries["REAL"] = True
                self.machines_real[subprob] = self.ml_real_creator(X_timeseries.shape[1:], len(dummy.columns))
                X_real = X_timeseries

            if np.sum(idx):
                if "sklearn" in str(type(self.machines_real[subprob])):
                    try:
                        self.machines_real[subprob].fit(
                            X_real[idx],
                            values.loc[idx],
                            sample_weight=np.ravel(self.row_weight.loc[idx])
                        )
                    except TypeError:
                        self.machines_real[subprob].fit(X_real[idx], values.loc[idx])
                else:
                    self.machines_real[subprob].fit(X_real[idx], values.loc[idx], epochs=epochs, verbose=verbose)

    def predict(self, labeled_data):
        """Predict data using supplied LabeledData.

        :param labeled_data: LabeledData. Dataset to predict. May have empty LabeledData.label.
        :return:             LabeledData. A copy of data, with LabeledData.pred filled in.
        """
        if not self.fitted:
            raise NotFittedError()

        data = labeled_data.normalize()

        X = data.data.values
        X_timeseries = data.to_timeseries()[0]
        y = pandas.DataFrame(index=data.data.index, columns=self.y_columns).fillna(0).astype(self.dtypes)
        y.loc[:, y.dtypes == np.object] = _NONE

        prim_dummy = self.dummies["CLASS"][_NONE]
        idx = pandas.MultiIndex.from_arrays([
            prim_dummy.columns.get_level_values(2), prim_dummy.columns.get_level_values(3)
        ])
        X_prim = X_timeseries if self.is_timeseries[_NONE] else X
        prim_smax = pandas.DataFrame(self.machines_class[""].predict(X_prim), columns=idx)
        for attr in ["danger_level", "emergency_warning"]:
            idx = np.argmax(prim_smax.loc[:, attr].values, axis=1)
            y["CLASS", "", attr] = np.array(list(prim_smax.loc[:, attr].columns))[idx]

        # Fix problem prediction to become a valid forecast (no double problems, no problem following _NONE).
        prob_cols = np.array([_NONE] + list(dict.fromkeys(PROBLEMS.values())))
        prob_smax = [pandas.DataFrame(prim_smax[f"problem_{n}"], columns=prob_cols).fillna(0) for n in [1, 2, 3]]
        prob_smax = np.stack(prob_smax, axis=1)
        idxs = np.flip(prob_smax.argsort(axis=2), axis=2)
        is_prob = np.any(np.sum(prob_smax.astype(np.bool), axis=2) > 1)
        for _ in [0, 1]:
            if is_prob:
                # If second likeliest problem_n-1 is very likely, use that instead
                fst = np.expand_dims(np.arange(y.shape[0]), axis=1)
                sec = [[0, 1]] * y.shape[0]
                likely = prob_smax[fst, sec, idxs[fst, sec, 1]] > 0.75 * prob_smax[fst, sec, idxs[fst, sec, 0]]
                idxs[:, 1:, 0] = idxs[:, 1:, 0] * np.invert(likely) + idxs[:, :-1, 1] * likely
                # If equal to problem_n-1/2, set to second likeliest alternative.
                prev_eq = idxs[:, 1:, :1] == idxs[:, :-1, :1]
                idxs[:, 1:, :-1] = idxs[:, 1:, :-1] * np.invert(prev_eq) + idxs[:, 1:, 1:] * prev_eq
            else:
                # If equal to problem_n-1/2, set to _NONE (as we only have 1-hot results).
                prev_eq = idxs[:, 1:, :1] == idxs[:, :-1, :1]
                idxs[:, 1:, :-1] = idxs[:, 1:, :-1] * np.invert(prev_eq)
            # Set to None if problem_n-1/2 was None.
            idxs[:, 1:] = idxs[:, 1:] * idxs[:, :-1, :1].astype(np.bool)
        for n in [1, 2, 3]:
            y["CLASS", "", f"problem_{n}"] = prob_cols[idxs[:, n - 1, 0]]
        y["CLASS", "", "problem_amount"] = np.sum(idxs[:, :, 0].astype(np.bool), axis=1).astype(np.int).astype("U")

        # Calculate relevant subproblems
        problem_cols = y.loc[:, [name.startswith("problem_") for name in y.columns.get_level_values(2)]]["CLASS", ""]
        for subprob, dummy in self.dummies["CLASS"].items():
            if subprob == _NONE:
                continue
            rows = np.any(np.char.equal(problem_cols.values.astype("U"), subprob), axis=1)
            if np.sum(rows):
                dummy = self.dummies["CLASS"][subprob]
                X_class = X_timeseries if self.is_timeseries["CLASS"] else X
                pred = pandas.DataFrame(
                    self.machines_class[subprob].predict(X_class[np.ix_(rows)]), columns=dummy.columns
                )
                for attr in pred.columns.get_level_values(2).unique():
                    idx = np.argmax(pred["CLASS", subprob].loc[:, attr].values, axis=1)
                    labels = np.array(list(pred["CLASS", subprob].loc[:, attr].columns))[idx]
                    y["CLASS", subprob, attr].values[np.ix_(rows)] = labels
        for subprob, dummy in self.dummies["MULTI"].items():
            rows = np.any(np.char.equal(problem_cols.values.astype("U"), subprob), axis=1)
            if np.sum(rows):
                dummy = self.dummies["MULTI"][subprob]
                X_multi = X_timeseries if self.is_timeseries["MULTI"] else X
                pred = pandas.DataFrame(
                    self.machines_multi[subprob].predict(X_multi[np.ix_(rows)]), columns=dummy.columns
                )
                for attr in pred.columns.get_level_values(2).unique():
                    labels = pred["MULTI", subprob, attr].values.astype(np.int).astype("U")
                    y["MULTI", subprob, attr] = "0"
                    y["MULTI", subprob, attr].values[np.ix_(rows)] = [''.join(row) for row in labels]
        for subprob, dummy in self.dummies["REAL"].items():
            rows = np.any(np.char.equal(problem_cols.values.astype("U"), subprob), axis=1)
            if np.sum(rows):
                dummy = self.dummies["REAL"][subprob]
                X_real = X_timeseries if self.is_timeseries["REAL"] else X
                pred = pandas.DataFrame(
                    self.machines_real[subprob].predict(X_real[np.ix_(rows)]), columns=dummy.columns
                )
                for attr in pred.columns.get_level_values(2).unique():
                    y["REAL", subprob, attr] = 0
                    y["REAL", subprob, attr].values[np.ix_(rows)] = pred["REAL", subprob, attr].values

        df = labeled_data.copy()
        df.pred = y
        return df


    def feature_importances(self):
        """Used to get all feature importances of internal classifiers.
        Supplied models must support model.feature_importances_, otherwise they are ignored.

        :return: DataFrame. Feature importances of internal classifiers.
        """
        importances = {}
        tupls = list(self.machines_class.items()) + list(self.machines_multi.items()) + list(self.machines_real.items())
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
        df = pandas.DataFrame(importances, index=self.X_columns)
        df.index.set_names(["feature_name", "day"], inplace=True)
        return df


    def dump(self, identifier):
        file_name = f'{se.local_storage}model_v{DILL_VERSION}_{identifier}.dill'
        tf_dict = {"CLASS": {}, "MULTI": {}, "REAL": {}}
        copy_class = self.machines_class.copy()
        copy_multi = self.machines_multi.copy()
        copy_real = self.machines_real.copy()

        for typ, d in ("CLASS", self.machines_class), ("MULTI", self.machines_multi), ("REAL", self.machines_real):
            for subprob, model in d.items():
                if "sklearn" in str(type(model)):
                    break
                model_filename = f"{file_name}_{typ}_{subprob}.h5"
                tf_dict[typ][subprob] = model_filename
                model.save(model_filename)
                d[subprob] = None

        with open(file_name, 'wb') as handle:
            dill.dump((self, tf_dict), handle)

        self.machines_class = copy_class
        self.machines_multi = copy_multi
        self.machines_real = copy_real

    @staticmethod
    def load(identifier):
        file_name = f'{se.local_storage}model_v{DILL_VERSION}_{identifier}.dill'
        with open(file_name, 'rb') as handle:
            (bm, tf_dict) = dill.load(handle)

        for typ, file_d in tf_dict.items():
            d = {"CLASS": bm.machines_class, "MULTI": bm.machines_multi, "REAL": bm.machines_real}[typ]
            for subprob, model_filename in file_d.items():
                d[subprob] = keras.load_model(model_filename)

        return bm


class Error(Exception):
    pass


class AlreadyFittedError(Error):
    pass


class NotFittedError(Error):
    pass


class FeatureImportanceMissingError(Error):
    pass

def prepare_class_weight_(class_weight, dummies):
    if class_weight is None:
        return None
    if class_weight == "balanced" or class_weight == "balanced_subsample":
        return class_weight

    prepared_weight = []
    for column in dummies.columns:
        typ, problem, attribute, label = column
        if attribute in class_weight and label in class_weight[attribute]:
            weight = class_weight[attribute][label]
            if len(dummies[typ, problem, attribute].columns) == 1:
                prepared_weight.append({1: weight[1]})
            else:
                prepared_weight.append(weight)
        else:
            if len(dummies[typ, problem, attribute].columns) == 1:
                prepared_weight.append({1: 1})
            else:
                prepared_weight.append({0: 1, 1: 1})
    return prepared_weight
