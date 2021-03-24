import re
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.machine.meta import setup, regobs_types
from avaml.machine.naive.naive_yesterday import NaiveYesterday
from avaml.machine.sk_clustered import SKClusteringMachine

from avaml.aggregatedata.__init__ import ForecastDataset, LabeledData, REG_ENG, CsvMissingError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import MultiTaskElasticNet

from avaml.machine.naive.naive_mode import NaiveMode
from avaml.machine.sk_classifier import SKClassifierMachine
from avaml.score import Score

seasons = ["2017-18", "2018-19", "2019-20"]
days = 1

try:
    print("Reading csv")
    labeled_data = LabeledData.from_csv(seasons=seasons, days=days, regobs_types=[], with_varsom=True)
except CsvMissingError:
    print("Csv missing. Fetching online data. (This takes a long time.)")
    labeled_data = ForecastDataset(seasons=seasons, regobs_types=[]).label(days=days, with_varsom=True)
    labeled_data.to_csv()

"""Naive Mode"""
for split, (training_data, testing_data, eval_data) in enumerate(labeled_data.split()):
    machine_id = "naive-mode"
    data_id = ""
    split_id = f"{machine_id}_{data_id}_split-{split}"

    print(f"Training {split_id}")
    bm = NaiveMode()
    bm.fit(training_data)

    bm.dump(split_id)

    print(f"Testing {split_id}")
    predicted_data = bm.predict(testing_data)
    f1 = predicted_data.f1()

    print("Writing F1 scores")
    f1.to_csv(f"{root}/output/{split_id}_f1.csv", sep=";")
    print("Writing distance scores")
    Score(predicted_data).calc().to_csv(f"{root}/output/{split_id}_scores.csv", sep=";")


"""Naive Previous Day"""
for split, (training_data, testing_data, eval_data) in enumerate(labeled_data.split()):
    machine_id = "naive-yesterday"
    data_id = ""
    split_id = f"{machine_id}_{data_id}_split-{split}"

    print(f"Training {split_id}")
    bm = NaiveYesterday()
    bm.fit(training_data)

    bm.dump(split_id)

    print(f"Testing {split_id}")
    predicted_data = bm.predict(testing_data)
    f1 = predicted_data.f1()

    print("Writing F1 scores")
    f1.to_csv(f"{root}/output/{split_id}_f1.csv", sep=";")
    print("Writing distance scores")
    Score(predicted_data).calc().to_csv(f"{root}/output/{split_id}_scores.csv", sep=";")

fds = {}
for days, varsom, regobs, noregions, nocause, temp, collapse, adam, fmt1, fmt4, levels, avy_idx in setup:
    if regobs and not avy_idx:
        regobs_avyidx = regobs
    elif regobs and avy_idx:
        regobs_avyidx = regobs + ["AvalancheIndex"]
    elif avy_idx:
        regobs_avyidx = regobs + ["AvalancheIndex"]
    else:
        regobs_avyidx = regobs
    if tuple(regobs_avyidx) not in fds:
        fds[tuple(regobs_avyidx)] = ForecastDataset(regobs_types=regobs_avyidx, seasons=seasons)
    labeled_data = fds[tuple(regobs_avyidx)].label(days=days, with_varsom=varsom)

    if noregions:
        labeled_data = labeled_data.drop_regions()
    if nocause:
        labeled_data.data = labeled_data.data.loc[
            :, [not re.search(r"cause", col) for col in labeled_data.data.columns.get_level_values(0)]
        ]
        labeled_data.scaler.fit(labeled_data.data)
    if temp:
        labeled_data = labeled_data.stretch_temperatures()
    if collapse:
        labeled_data = labeled_data.to_time_parameters(orig_days=1)
    labeled_data = labeled_data.normalize()

    data_id = f"days-{days}_{'no' if noregions else ''}regions_{'' if varsom else 'no'}varsom_{'-'.join(regobs_avyidx)}"
    data_id += f"{'_nocause' if nocause else ''}{'_temp' if temp else ''}{'_collapsed' if collapse else ''}"
    data_id += f"{'_adam' if adam else ''}"
    data_id += f"{'_fmt1' if fmt1 else ''}{'_fmt4' if fmt4 else ''}{'_levels' if levels else ''}"

    for split, (training_data, testing_data, eval_data) in enumerate(labeled_data.split()):
        if fmt4:
            training_data = training_data.to_elevation_fmt_4()
            testing_data = testing_data.to_elevation_fmt_4(exclude_label=True)
        if fmt1:
            training_data = training_data.to_elevation_fmt_1()
            testing_data = testing_data.to_elevation_fmt_1(exclude_label=True)
        if levels:
            training_data = training_data.to_elev_class()
            testing_data = testing_data.to_elev_class(exclude_label=True)

        """SKClassifier Decisiontree"""
        machine_id = "skclassifier-decisiontree"
        split_id = f"{machine_id}_{data_id}_split-{split}"

        print(f"Training {split_id}")
        def classifier_creator(indata, outdata, class_weight=None):
            return DecisionTreeClassifier()

        def regressor_creator(indata, outdata):
            return DecisionTreeRegressor()

        bm = SKClassifierMachine(
            classifier_creator,
            classifier_creator,
            classifier_creator,
            regressor_creator,
        )
        bm.fit(training_data)

        bm.dump(split_id)

        print(f"Testing {split_id}")
        predicted_data = bm.predict(testing_data)
        if levels:
            predicted_data = predicted_data.from_elev_class()
        f1 = predicted_data.f1()

        print("Writing F1 scores")
        f1.to_csv(f"{root}/output/{split_id}_f1.csv", sep=";")
        print("Writing distance scores")
        Score(predicted_data).calc().to_csv(f"{root}/output/{split_id}_scores.csv", sep=";")

        """SKClustering default"""
        machine_id = "skclustering-default"
        split_id = f"{machine_id}_{data_id}_split-{split}"

        print(f"Training {split_id}")
        dt = DecisionTreeClassifier()
        clustering = AgglomerativeClustering(n_clusters=20)
        bm = SKClusteringMachine(dt, clustering)
        bm.fit(training_data)

        bm.dump(split_id)

        print(f"Testing {split_id}")
        predicted_data = bm.predict(testing_data)
        if levels:
            predicted_data = predicted_data.from_elev_class()
        f1 = predicted_data.f1()

        print("Writing F1 scores")
        f1.to_csv(f"{root}/output/{split_id}_f1.csv", sep=";")
        print("Writing distance scores")
        Score(predicted_data).calc().to_csv(f"{root}/output/{split_id}_scores.csv", sep=";")

#def mean(machine_id, data_id=""):
#    if data_id:
#        (days, varsom, regobs, noregions, temp, cause, collapse, adam, levels, class1, avalancheidx) = data_id
#        if avalancheidx:
#            regobs_avyidx = regobs + ['AvalancheIndex']
#        else:
#            regobs_avyidx = regobs
#
#        data_id = f"days-{days}_{'no' if noregions else ''}regions_{'' if varsom else 'no'}varsom_{'-'.join(regobs_avyidx)}"
#        data_id += f"{'_temp' if temp else ''}{'' if cause else '_nocause'}{'_collapsed' if collapse else ''}"
#        data_id += f"{'_adam' if adam else ''}{'_levels' if levels else ''}{'_class1' if class1 else ''}"
#    else:
#        (days, varsom, regobs, noregions, temp, cause, collapse, adam, levels, class1, avalancheidx) =\
#            (1, True, [], True, False, False, False, False, False, False, False)
#
#    print(data_id)
#
#    id = f"{machine_id}_{data_id}"
#    df = None
#    for split in range(0, 3):
#        row = (machine_id, days, varsom, bool(regobs), noregions, temp, cause, collapse, adam, levels, class1, avalancheidx, split)
#
#        split_id = f"{id}_split-{split}"
#        path_scores = f"{root}/output/{split_id}_scores.csv"
#        scores = pd.read_csv(path_scores, sep=";", header=[0, 1], index_col=[0, 1])
#
#        names = [
#            "machine",
#            "days",
#            "varsom",
#            "regobs",
#            "noregions",
#            "temp",
#            "cause",
#            "collapse",
#            "adam",
#            "levels",
#            "class1",
#            "avalancheidx",
#            "split",
#        ]
#        index = pd.MultiIndex.from_arrays([[], [], [], [], [], [], [], [], [], [], [], [], []], names=names)
#        columns = [(re.sub(r'Unnamed:.*', "", col[0]), col[1]) for col in scores.columns.tolist()]
#
#        scores.columns = pd.MultiIndex.from_tuples(columns)
#        average = scores.abs().mean(axis=0).to_frame().values.T
#        average = pd.DataFrame(average, index=pd.MultiIndex.from_tuples([row], names=names), columns=scores.columns)
#        if df is None:
#            df = pd.DataFrame(index=index, columns=scores.columns)
#        df = pd.concat([df, average])
#    return df
#
#distances = pd.DataFrame()
#
##distances = pd.concat([distances, mean("naive-mode")])
##distances = pd.concat([distances, mean("naive-yesterday")])
##distances = pd.concat([distances, mean("skclassifier-neural", (10, True, [], False, False, False, False, True, False))])
##distances = pd.concat([distances, mean("skclassifier-neural", (10, True, [], False, False, False, False, True, True))])
##distances = pd.concat([distances, mean("skclassifier-neural", (10, True, [], True, False, False, False, False, True, False))])
##distances = pd.concat([distances, mean("skclassifier-neural", (10, True, [], False, False, False, False, False, True, False))])
##distances = pd.concat([distances, mean("skclassifier-neural", (10, False, [], True, False, False, False, False, False, False))])
##distances = pd.concat([distances, mean("skclassifier-neural", (10, True, [], True, False, False, False, False, False, False))])
##distances = pd.concat([distances, mean("skclassifier-neural", (10, True, [], True, False, False, False, True, False, False))])
##distances = pd.concat([distances, mean("skclassifier-default", (10, False, [], True, False, False, False, True, False, False))])
##distances = pd.concat([distances, mean("skclassifier-default", (10, True, [], True, False, False, False, True, False, False))])
##distances = pd.concat([distances, mean("skclustering-default", (10, False, [], True, False, False, False, True, False, False))])
##distances = pd.concat([distances, mean("skclustering-default", (10, True, [], True, False, False, False, True, False, False))])
#distances = pd.concat([distances, mean("skclassifier-default", (10, True, [], True, False, False, False, False, False, True, False))])
#
##for days, varsom, regobs, temp in setup:
#    #for levels, avy_idx in [(True, True), (True, False), (False, True)]:
##    distances = pd.concat([distances, mean("skclassifier-default", (days, varsom, regobs, False, False, False, False, False, False, False, False))])
#
#for days, varsom, regobs, temp in setup:
#    #for cause, collapse in [(True, True), (True, False), (False, True), (False, False)]:
#    #    if collapse and days <= 2:
#    #        continue
#
#    data_id = (days, varsom, regobs, True, temp, False, False, False, False, False, False)
#
#    distances = pd.concat([distances, mean("skclassifier-default", data_id)])
#    distances = pd.concat([distances, mean("skclustering-default", data_id)])
#
#distances.loc[:, :] = distances.values.astype(np.float)
#distances = distances.mean(axis=0, level=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).sort_values(by=("", "distance"))
#distances.to_csv(f"{root}/output/combined_scores3.csv", sep=";")
#distances = distances.loc[distances.index.get_level_values(2) == True]
#print(distances)
#print(distances.index.to_frame())
#print(pd.DataFrame(np.arange(distances.shape[0]).reshape((distances.shape[0], 1))))
#print(distances.index.to_frame())
#print(distances[[("", "distance")]])
#
#distances = distances.loc["skclassifier-default"]
#opt_distance = pd.concat([
#    pd.DataFrame(np.arange(distances.shape[0]).reshape((distances.shape[0], 1)), index=distances.index),
#    distances.index.to_frame(),
#    distances[("", "distance")].rename("distance"),
#], axis=1).apply(np.float)
#
#print(opt_distance)


#n_bins = 40
#axis = (.065, .075)
#fig, axs = plt.subplots(7-3, 2, sharey=True, tight_layout=True)
#for n, level in enumerate(range(3, 7)):
#    axs[n][0].hist(distances.loc[distances.index.get_level_values(level) == True, ("", "distance")], bins=n_bins)
#    axs[n][1].hist(distances.loc[distances.index.get_level_values(level) == False, ("", "distance")], bins=n_bins)
#    axs[n][0].set(xlim=axis)
#    axs[n][1].set(xlim=axis)
#plt.show()
