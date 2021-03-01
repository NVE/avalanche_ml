from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

import sys
import os

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.aggregatedata.__init__ import CsvMissingError, LabeledData, ForecastDataset
from avaml.machine.sk_classifier import SKClassifierMachine
from avaml.machine.sk_clustered import SKClusteringMachine
from analysis.generate_setups import setup

train_seasons = ["2018-19", "2019-20"]
test_seasons = ["2017-18"]

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

f1 = None
for days, varsom, regobs in setup:
    try:
        labeled_data = LabeledData.from_csv(days=days, regobs_types=regobs, with_varsom=varsom)
    except CsvMissingError:
        labeled_data = ForecastDataset(regobs_types=regobs).label(days=days, with_varsom=varsom)
        labeled_data.to_csv()

    labeled_data = labeled_data.normalize()
    labeled_data = labeled_data.drop_regions()

    for m_tag, create_machine in [("SKClustering", createClustering), ("SKClassifier", createClassifier)]:
        tag = f"{m_tag}_{days}_noregions_{'' if varsom else 'no'}varsom_{'-'.join(regobs)}"
        print(f"Training {tag}, size {labeled_data.data.shape}")

        machine = create_machine()
        machine.fit(labeled_data)
        print("Saving machine")
        machine.dump(tag)

        results_machine = None
        strat = ("CLASS", "", "danger_level")
        for split_idx, (training_data, testing_data) in enumerate(labeled_data.kfold(5, stratify=strat)):
            print(f"Training fold: {split_idx}")
            machine = create_machine()
            machine.fit(training_data)

            print(f"Testing fold: {split_idx}")
            predicted_data = machine.predict(testing_data)
            results_series = predicted_data.f1()
            results_machine = results_series if results_machine is None else results_machine + (results_series - results_machine) / (split_idx + 1)
            if results_machine is None:
                results_machine = results_series
            else:
                results_machine = results_machine + (results_series - results_machine) / (split_idx + 1)

        f1_machine = results_machine["f1"].rename(tag)
        f1 = f1_machine if f1 is None else pd.concat([f1, f1_machine], axis=1)
print("Saving results")
f1.to_csv(f"{root}/output/results_trained_models.csv", sep=";")