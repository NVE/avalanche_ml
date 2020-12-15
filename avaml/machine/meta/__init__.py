import dill
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

from avaml import Error, setenvironment as se
from avaml.aggregatedata import CsvMissingError, LabeledData, ForecastDataset, NoBulletinWithinRangeError, \
    DatasetMissingLabel, NoDataFoundError
from avaml.machine.sk_classifier import SKClassifierMachine
from avaml.machine.sk_clustered import SKClusteringMachine
from avaml.machine.meta.generate_setups import setup, regobs_types
from avaml.machine import DILL_VERSION, AlreadyFittedError

expected_errors = (NoBulletinWithinRangeError, DatasetMissingLabel, NoDataFoundError)

def createClustering():
    dt = DecisionTreeClassifier(max_depth=7, class_weight={})
    clustering = AgglomerativeClustering(n_clusters=20)
    return SKClusteringMachine(dt, clustering)


def createClassifier():
    def classifier_creator(indata, outdata, class_weight=None):
        return DecisionTreeClassifier(max_depth=7, class_weight=class_weight)

    def regressor_creator(indata, outdata):
        return MultiTaskElasticNet()

    return SKClassifierMachine(
        classifier_creator,
        classifier_creator,
        classifier_creator,
        regressor_creator,
        sk_prim_class_weight={},
        sk_class_weight={},
    )

class MetaMachine:
    def __init__(self):
        self.machines = {}
        self.f1 = None
        self.fitted = False

    def fit(self, seasons=('2017-18', '2018-19', '2019-20')):
        if self.fitted:
            raise AlreadyFittedError()
        self.fitted = True
        fd_noregobs = ForecastDataset(regobs_types=[], seasons=seasons)
        fd_regobs = ForecastDataset(regobs_types=regobs_types, seasons=seasons)

        for days, varsom, regobs in setup:
            if regobs:
                labeled_data = fd_regobs.label(days=days, with_varsom=varsom)
            else:
                labeled_data = fd_noregobs.label(days=days, with_varsom=varsom)

            labeled_data = labeled_data.normalize()
            labeled_data = labeled_data.drop_regions()

            for m_tag, create_machine in [("SKClustering", createClustering), ("SKClassifier", createClassifier)]:
                tag = f"{m_tag}_{days}_noregions_{'' if varsom else 'no'}varsom_{'-'.join(regobs)}"
                print(f"Training {tag}, size {labeled_data.data.shape}")

                machine = create_machine()
                machine.fit(labeled_data)
                print("Saving machine")
                self.machines[tag] = machine
                #machine.dump(tag)

                results_machine = None
                strat = ("CLASS", "", "danger_level")
                for split_idx, (training_data, testing_data) in enumerate(labeled_data.kfold(5, stratify=strat)):
                    print(f"Training fold: {split_idx}")
                    machine = create_machine()
                    machine.fit(training_data)

                    print(f"Testing fold: {split_idx}")
                    predicted_data = machine.predict(testing_data)
                    results_series = predicted_data.f1()
                    results_machine = results_series if results_machine is None else results_machine + (
                            results_series - results_machine) / (split_idx + 1)
                    if results_machine is None:
                        results_machine = results_series
                    else:
                        results_machine = results_machine + (results_series - results_machine) / (split_idx + 1)

                f1_machine = results_machine["f1"].rename(tag)
                self.f1 = f1_machine if self.f1 is None else pd.concat([self.f1, f1_machine], axis=1)

    def predict(self, seasons=["2020-21"], csv_tag=None):
        if csv_tag is None:
            fd_noregobs = ForecastDataset(regobs_types=[], seasons=seasons)
            fd_regobs = ForecastDataset(regobs_types=regobs_types, seasons=seasons)
        all_data = {}
        machine_scores = self.f1
        ms_idx = machine_scores.index.to_frame().fillna("")
        machine_scores.index = pd.MultiIndex.from_frame(ms_idx)
        machine_scores = machine_scores.loc[["CLASS"]]
        empty_indices = machine_scores.index[np.logical_or(
            machine_scores.index.get_level_values(3) == "0", machine_scores.index.get_level_values(3) == ""
        )]
        machine_scores = machine_scores.drop(empty_indices)
        grouped_scores = machine_scores.groupby(level=[0, 1, 2]).mean()

        for days, varsom, regobs in setup:
            d_tag = f"{days}_noregions_{'' if varsom else 'no'}varsom_{'-'.join(regobs)}"
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
                if pred is not None and label in pred.columns:
                    combined = pred[label].combine_first(pred_tag)
                    pred = pred.reindex(pred.index.union(combined.index))
                    pred.loc[combined.index, label] = combined
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


