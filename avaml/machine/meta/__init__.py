import os
import re

import dill
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

from avaml import setenvironment as se
from avaml.aggregatedata.__init__ import LabeledData, ForecastDataset, NoBulletinWithinRangeError, \
    DatasetMissingLabel, NoDataFoundError
from avaml.machine.sk_classifier import SKClassifierMachine
from avaml.machine.sk_clustered import SKClusteringMachine
from avaml.machine.meta.generate_setups import setup, regobs_types
from avaml.machine import DILL_VERSION, AlreadyFittedError

expected_errors = (NoBulletinWithinRangeError, DatasetMissingLabel, NoDataFoundError)

root = f"{os.path.dirname(os.path.abspath(__file__))}/../../.."

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
        self.f1 = None
        self.fitted = False
        self.with_varsom = with_varsom
        self.stretch_temp = stretch_temp

    def fit(self, seasons=['2019-20'], season_train='2018-19'):
        if self.fitted:
            raise AlreadyFittedError()
        self.fitted = True
        fd_noregobs = ForecastDataset(regobs_types=[], seasons=seasons)
        fd_regobs = ForecastDataset(regobs_types=regobs_types, seasons=seasons)
        fd_noregobs_test = ForecastDataset(regobs_types=[], seasons=[season_train])
        fd_regobs_test = ForecastDataset(regobs_types=regobs_types, seasons=[season_train])

        for days, varsom, regobs, temp in setup:
            if varsom and not self.with_varsom:
                continue
            if temp and self.stretch_temp is not None and not self.stretch_temp:
                continue
            if not temp and self.stretch_temp:
                continue

            if regobs:
                labeled_data = fd_regobs.label(days=days, with_varsom=varsom)
                test_data = fd_regobs_test.label(days=days, with_varsom=varsom)
            else:
                labeled_data = fd_noregobs.label(days=days, with_varsom=varsom)
                test_data = fd_noregobs_test.label(days=days, with_varsom=varsom)

            labeled_data.data = labeled_data.data.loc[
                 :, [not re.search(r"cause", col) for col in labeled_data.data.columns.get_level_values(0)]
            ]
            test_data.data = test_data.data.loc[
                :, [not re.search(r"cause", col) for col in test_data.data.columns.get_level_values(0)]
            ]
            if temp:
                labeled_data = labeled_data.stretch_temperatures()
                test_data = test_data.stretch_temperatures()
            labeled_data = labeled_data.drop_regions()
            test_data = test_data.drop_regions()
            if days > 2:
                labeled_data = labeled_data.to_time_parameters(orig_days=1)
                test_data = test_data.to_time_parameters(orig_days=1)
            labeled_data = labeled_data.normalize()
            test_data = test_data.normalize(by=labeled_data)

            for m_tag, create_machine in [("SKClustering", createClustering), ("SKClassifier", createClassifier)]:
                tag = f"{m_tag}_{days}_noregions_{'' if varsom else 'no'}varsom_{'-'.join(regobs)}{'_temp' if temp else ''}"
                print(f"Training {tag}, size {labeled_data.data.shape}")

                machine = create_machine()
                machine.fit(labeled_data)
                print("Saving machine")
                self.machines[tag] = machine

                print("Testing machine")
                predicted_data = machine.predict(test_data)
                results_machine = predicted_data.f1()

                real = results_machine.index.get_level_values(0) == "REAL"
                # We need to "reverse" the rmse to be sorted together with f1.
                results_machine.loc[real, "rmse"] = results_machine\
                    .loc[real, "rmse"]\
                    .rdiv(1, fill_value=0)\
                    .replace(np.inf, 0)
                f1_machine = results_machine[["f1", "rmse"]]\
                    .apply(lambda x: pd.Series(x.dropna().to_numpy()), axis=1)\
                    .squeeze()\
                    .rename(tag)
                self.f1 = f1_machine if self.f1 is None else pd.concat([self.f1, f1_machine], axis=1).fillna(0)

    def predict(self, seasons=["2020-21"], csv_tag=None):
        if csv_tag is None:
            fd_noregobs = ForecastDataset(regobs_types=[], seasons=seasons)
            fd_regobs = ForecastDataset(regobs_types=regobs_types, seasons=seasons)
        all_data = {}
        machine_scores = self.f1
        ms_idx = machine_scores.index.to_frame().fillna("")
        machine_scores.index = pd.MultiIndex.from_frame(ms_idx)
        empty_indices = machine_scores.index[np.logical_and(
            np.logical_or(
                machine_scores.index.get_level_values(3) == "0", machine_scores.index.get_level_values(3) == ""
            ),
            machine_scores.index.get_level_values(0) != "REAL"
        )]
        machine_scores = machine_scores.drop(empty_indices)
        groupby = machine_scores.groupby(level=[0, 1, 2])
        grouped_scores = groupby.mean() + groupby.min()

        for days, varsom, regobs, temp in setup:
            if varsom and not self.with_varsom:
                continue
            if temp and self.stretch_temp is not None and not self.stretch_temp:
                continue
            if not temp and self.stretch_temp:
                continue

            d_tag = f"{days}_noregions_{'' if varsom else 'no'}varsom_{'-'.join(regobs)}{'_temp' if temp else ''}"
            print(d_tag)
            try:
                print("Collecting data")
                if csv_tag is None:
                    fd = fd_regobs if regobs else fd_noregobs
                    data = fd.label(days=days, with_varsom=varsom)
                else:
                    data = LabeledData.from_csv(days, regobs, False, varsom, csv_tag)
                data = data.normalize()
                data = data.drop_regions()
                if temp:
                    data = data.stretch_temperatures()

                collected = True
            except expected_errors:
                print("Failed to collect data")
                collected = False
            for m_tag, machine_class in [("SKClustering", SKClusteringMachine), ("SKClassifier", SKClassifierMachine)]:
                tag = f"{m_tag}_{d_tag}"
                if collected:
                    machine = self.machines[tag]
                    all_data[tag] = data
                else:
                    grouped_scores.drop(columns=tag, inplace=True)

        predictions = {}
        best_models = pd.DataFrame(
            grouped_scores.columns.values[np.argsort(-grouped_scores)],
            index=grouped_scores.index
        )

        ld = None
        for tag in np.unique(best_models.values.flatten()):
            print(tag)
            labeled_data = self.machines[tag].predict(all_data[tag], force_subprobs=True)
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


