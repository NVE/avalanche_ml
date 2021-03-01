from datetime import date

import pandas as pd
import numpy as np
import sys
import os

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.aggregatedata.__init__ import ForecastDataset, NoBulletinWithinRangeError, DatasetMissingLabel, NoDataFoundError
from avaml.machine.sk_classifier import SKClassifierMachine
from avaml.machine.sk_clustered import SKClusteringMachine
from analysis.generate_setups import setup, regobs_types

pickle_file = f"{root}/output/meta_test_data_.pickle"

machine_scores = pd.read_csv(f"{root}/output/results_trained_models.csv", sep=";", index_col=[0, 1, 2, 3], header=[0])
ms_idx = machine_scores.index.to_frame().fillna("")
machine_scores.index = pd.MultiIndex.from_frame(ms_idx)
machine_scores = machine_scores.loc[["CLASS"]]
empty_indices = machine_scores.index[np.logical_or(
    machine_scores.index.get_level_values(3) == "0", machine_scores.index.get_level_values(3) == ""
)]
machine_scores.drop(empty_indices, inplace=True)
grouped_scores = machine_scores.groupby(level=[0, 1, 2]).mean()

tomorrow = date.fromisoformat("2019-03-31")#date.today() + timedelta(days=1)
expected_errors = (NoBulletinWithinRangeError, DatasetMissingLabel, NoDataFoundError)
today_data = {}
machines = {}
try:
    raise FileNotFoundError
    with open(pickle_file, 'rb') as handle:
        fd_noregobs, fd_regobs = dill.load(handle)
except FileNotFoundError:
    fd_noregobs = ForecastDataset(regobs_types=[], seasons=["2020-21"])
    fd_regobs = ForecastDataset(regobs_types=regobs_types, seasons=["2020-21"])
    #with open(pickle_file, 'wb') as handle:
    #    dill.dump((fd_noregobs, fd_regobs), handle)
for days, varsom, regobs in setup:
    d_tag = f"{days}_noregions_{'' if varsom else 'no'}varsom_{'-'.join(regobs)}"
    print(d_tag)
    try:
        print("Collecting data")
        fd = fd_regobs if regobs else fd_noregobs
        data = fd.label(days=days, with_varsom=varsom)
        # data = ForecastDataset\
        #    .date(regobs_types=regobs, date=tomorrow, days=days, use_label=False)\
        #    .label(days=days, with_varsom=varsom)
        data = data.normalize()
        data = data.drop_regions()
        collected=True
    except expected_errors:
        print("Failed to collect data")
        collected = False
    for m_tag, machine_class in [("SKClustering", SKClusteringMachine), ("SKClassifier", SKClassifierMachine)]:
        tag = f"{m_tag}_{d_tag}"
        if collected:
            machine = machine_class.load(tag)
            machines[tag] = machine
            today_data[tag] = data
        else:
            grouped_scores.drop(columns=tag, inplace=True)

predictions = {}
best_models = pd.DataFrame(
    grouped_scores.columns.values[np.argsort(-grouped_scores)],
    index=grouped_scores.index
)
best_models.to_csv(f"{root}/output/best_models.csv", sep=";")
ld = None
for tag in np.unique(best_models.values.flatten()):
    print(tag)
    labeled_data = machines[tag].predict(today_data[tag], force_subprobs=True)
    predictions[tag] = labeled_data.pred
    if ld is None:
        ld = labeled_data
        ld.data = None
    else:
        combined = ld.label.combine_first(labeled_data.label)
        ld.label = ld.label.reindex(ld.label.index.union(labeled_data.label.index))
        ld.label.loc[combined.index] = combined

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
ld = ld.valid_pred()

pred.to_csv(f"{root}/output/meta_mean_pred_2021.csv", sep=";")

ld.f1().to_csv(f"{root}/output/meta_mean_f1_2021.csv", sep=";")
