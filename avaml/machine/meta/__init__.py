import os
import re

import dill
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import numpy as np

from avaml import setenvironment as se, Error
from avaml.aggregatedata.__init__ import LabeledData, ForecastDataset, NoBulletinWithinRangeError, \
    DatasetMissingLabel, NoDataFoundError
from avaml.machine.sk_classifier import SKClassifierMachine
from avaml.machine.sk_clustered import SKClusteringMachine
from avaml.machine.meta.generate_setups import setup, regobs_types, get_data
from avaml.machine import DILL_VERSION, AlreadyFittedError
from avaml.score import Score, WET_WEIGHT, LOOSE_WEIGHT

expected_errors = (NoBulletinWithinRangeError, DatasetMissingLabel, NoDataFoundError)

root = f"{os.path.dirname(os.path.abspath(__file__))}/../../.."

SCORE_MAP = {
    "danger_level": [("global", "danger_level")],
    "emergency_warning": [("global", "emergency_warning")],
    "problem_1": [("problem_1", "loose"), ("problem_1", "wet")],
    "problem_2": [("problem_1", "loose"), ("problem_1", "wet")],
    "problem_3": [("problem_1", "loose"), ("problem_1", "wet")],
    "problem_amount": [("problem_1", "loose"), ("problem_1", "wet")],
    "cause": [("problem_1", "freq")],
    "dist": [("problem_1", "freq")],
    "dsize": [("problem_1", "freq")],
    "prob": [("problem_1", "freq")],
    "trig": [("problem_1", "freq")],
    "aspect": [("problem_1", "spatial_diff")],
    "lev_fill": [("problem_1", "spatial_diff")],
    "lev_min": [("problem_1", "spatial_diff")],
    "lev_max": [("problem_1", "spatial_diff")],
}

F1_MAP = [
    ("CLASS", "", "danger_level", "4"),
    ("CLASS", "", "emergency_warning", "Naturlig utløste skred"),
    ("CLASS", "", "problem_1", "pwl-slab"),
]


def createClustering():
    dt = DecisionTreeClassifier(max_depth=7, class_weight={})
    clustering = AgglomerativeClustering(n_clusters=20)
    return SKClusteringMachine(dt, clustering)


def createClassifier():
    def classifier_creator(indata, outdata, class_weight=None):
        return DecisionTreeClassifier(max_depth=7, class_weight=class_weight)

    def regressor_creator(indata, outdata):
        return MultiTaskElasticNet(max_iter=3000)

    return SKClassifierMachine(
        classifier_creator,
        classifier_creator,
        classifier_creator,
        regressor_creator,
        sk_prim_class_weight={},
        sk_class_weight={},
    )

class MetaMachine:
    def __init__(self, with_varsom=True, stretch_temp=None):
        self.machines = {}
        self.scores = None
        self.fitted = False
        self.with_varsom = with_varsom
        self.stretch_temp = stretch_temp

    def fit(self, machines, seasons=['2017-18', '2018-19', '2019-20']):
        if self.fitted:
            raise AlreadyFittedError()
        self.seasons = seasons
        self.fitted = True
        self.machines = {}
        self.machines_conf = {}

        for machine in machines:
            type, days, varsom, regobs, noregions, nocause, temp, collapse, adam, fmt1, fmt4, levels, avy_idx = machine
            if type in ["skclassifier-decisiontree", "skclassifier-neural"]:
                bm_class = SKClassifierMachine
            elif type in ["skclustering-default"]:
                bm_class = SKClusteringMachine
            else:
                raise UnknownMachineId()

            ld, data_id = get_data(
                seasons,
                days,
                varsom,
                regobs,
                noregions,
                nocause,
                temp,
                collapse,
                adam,
                fmt1,
                fmt4,
                levels,
                avy_idx
            )

            for split, (training_data, test_data, _) in enumerate(ld.split()):
                if fmt4:
                    test_data = test_data.to_elevation_fmt_4(exclude_label=True)
                if fmt1:
                    test_data = test_data.to_elevation_fmt_1(exclude_label=True)
                if levels:
                    test_data = test_data.to_elev_class(exclude_label=True)
                split_id = f"{type}_{data_id}_split-{split}"
                print(f"Loading machine: {split_id}")
                bm = bm_class.load(split_id)
                self.machines[split_id] = bm
                self.machines_conf[split_id] = machine[1:]

                print("Testing machine")
                predicted_data = bm.predict(test_data)
                if levels:
                    predicted_data = predicted_data.from_elev_class()
                if adam:
                    predicted_data = predicted_data.adam()
                columns = predicted_data.pred.columns
                scores_ = Score(predicted_data).calc()
                #f1_ = predicted_data.f1().loc[F1_MAP, "f1"]
                #path_scores = f"{root}/output/{split_id}_scores.csv"
                #scores_ = pd.read_csv(path_scores, sep=";", header=[0, 1], index_col=[0, 1])
                scores = scores_.abs().mean(axis=0).to_frame().values
                scores = pd.DataFrame(scores, index=scores_.columns, columns=[split_id])
                mapped_scores = pd.Series(index=columns, name=split_id)
                for column in columns:
                    idx = mapped_scores.index.get_level_values(2) == column[2]
                    if column[2] in ["problem_1", "problem_2", "problem_3", "problem_amount"]:
                        average = np.mean(scores.loc[SCORE_MAP[column[2]]].values * [WET_WEIGHT, LOOSE_WEIGHT])
                    else:
                        average = scores.loc[SCORE_MAP[column[2]]].mean().iloc[0]
                    mapped_scores.loc[idx] = average
                #results_machine = predicted_data.f1()

                #real = results_machine.index.get_level_values(0) == "REAL"
                ## We need to "reverse" the rmse to be sorted together with f1.
                #results_machine.loc[real, "rmse"] = results_machine \
                #    .loc[real, "rmse"] \
                #    .rdiv(1, fill_value=0) \
                #    .replace(np.inf, 0)

                #nreal = np.logical_not(real)
                ## Multiply f1 with share
                #results_machine.loc[nreal, "f1"] = results_machine.loc[nreal, "precision"] * results_machine.loc[nreal, "share"]

                #f1_machine = results_machine[["f1", "rmse"]] \
                #    .apply(lambda x: pd.Series(x.dropna().to_numpy()), axis=1) \
                #    .squeeze() \
                #    .rename(split_id)
                self.scores = mapped_scores if self.scores is None else pd.concat([self.scores, mapped_scores], axis=1).fillna(0)
        self.scores.to_csv("/home/aron/Downloads/mapped_scores.csv", sep=";")

    def predict(self, regions=[3011, 3016, 3035], seasons=None):
        seasons = seasons if seasons is not None else self.seasons
        machine_scores = self.scores
        #ms_idx = machine_scores.index.to_frame().fillna("")
        #machine_scores.index = pd.MultiIndex.from_frame(ms_idx)
        #empty_indices = machine_scores.index[np.logical_and(
        #    np.logical_or(
        #        machine_scores.index.get_level_values(3) == "0", machine_scores.index.get_level_values(3) == ""
        #    ),
        #    machine_scores.index.get_level_values(0) != "REAL"
        #)]
        #machine_scores = machine_scores.drop(empty_indices)
        #groupby = machine_scores.groupby(level=0)
        #grouped_scores = groupby.sum()

        predictions = {}
        best_models = pd.DataFrame(
            machine_scores.columns.values[np.argsort(machine_scores, axis=1)],
            index=machine_scores.index
        )
        best_models.to_csv("/home/aron/Downloads/best_models.csv", sep=";")

        all_data = {}
        for tag in np.unique(best_models.values.flatten()):
            if tag not in all_data:
                ld = get_data(seasons, *self.machines_conf[tag])[0]
                ld.data = ld.data.iloc[[region in regions for region in ld.data.index.get_level_values(1)]]
                ld.label = ld.label.iloc[[region in regions for region in ld.label.index.get_level_values(1)]]
                ld.pred = ld.pred.iloc[[region in regions for region in ld.pred.index.get_level_values(1)]]
                ld.row_weight = ld.row_weight.iloc[
                    [region in regions for region in ld.row_weight.index.get_level_values(1)]]
                if self.machines_conf[tag][-3]:
                        test_data = test_data.to_elevation_fmt_4(exclude_label=True)
                if self.machines_conf[tag][-4]:
                    test_data = test_data.to_elevation_fmt_1(exclude_label=True)
                if self.machines_conf[tag][-2]:
                    test_data = test_data.to_elev_class(exclude_label=True)
                all_data[tag] = ld

        ld = None
        for tag in np.unique(best_models.values.flatten()):
            print(tag)
            labeled_data = self.machines[tag].predict(all_data[tag], force_subprobs=True)
            if self.machines_conf[tag][-2]:
                labeled_data = labeled_data.from_elev_class()
            if self.machines_conf[tag][-5]:
                labeled_data = labeled_data.adam()
            #labeled_data = labeled_data.to_elevation_fmt_4()
            labeled_data.pred = labeled_data.pred.astype(str)
            predictions[tag] = labeled_data.pred
            if ld is None:
                ld = labeled_data
                ld.data = None
            elif ld.label is not None and labeled_data.label is not None:
                combined = ld.label.combine_first(labeled_data.label)
                ld.label = ld.label.reindex(ld.label.index.union(labeled_data.label.index))
                ld.label.loc[combined.index] = combined
            elif ld.label is None and labeled_data.label is not None:
                ld.label = labeled_data.label

        pred = None
        for label in best_models.index:
            for _, tag in best_models.loc[label].items():
                pred_tag = predictions[tag][label].replace("0", np.nan)
                if label[1] != "":
                    pred_tag = pred_tag.replace("", np.nan)
                if pred is not None and label in pred.columns:
                    combined = pred[label].combine_first(pred_tag)
                    pred = pred.reindex(combined.index)
                    pred[label] = combined
                elif pred is not None:
                    pred = pred.reindex(pred.index.union(pred_tag.index))
                    pred.loc[predictions[tag].index, label] = pred_tag
                else:
                    pred = pred_tag.to_frame()

        """Remove values that shouldn't exist."""
        ld.pred = pred
        return ld.valid_pred()

    def dump(self, identifier):
        file_name = f'{se.local_storage}model_meta_v{DILL_VERSION}_{identifier}.dill'

        with open(file_name, 'wb') as handle:
            dill.dump(self, handle)

    @staticmethod
    def load(identifier):
        file_name = f'{se.local_storage}model_meta_v{DILL_VERSION}_{identifier}.dill'
        with open(file_name, 'rb') as handle:
            bm = dill.load(handle)
        return bm


class UnknownMachineId(Error):
    pass
