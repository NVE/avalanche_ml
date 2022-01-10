import re

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.tree import DecisionTreeClassifier

import sys
import os

from avaml.aggregatedata import LabeledData, CsvMissingError, ForecastDataset

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.machine.sk_classifier import SKClassifierMachine
from avaml.machine.sk_clustered import SKClusteringMachine
def createClustering():
    dt = DecisionTreeClassifier(max_depth=7)
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

regobs_types = [
    "Faretegn",
    "Tester",
    "Skredaktivitet",
    "Snødekke",
    "Skredproblem",
    "Skredfarevurdering"
]

secondary_variations = [
    (False, False, False, False, False, False, False, False, False),
    (True, False, False, False, False, False, False, False, False),
    (False, True, False, False, False, False, False, False, False),
    (False, False, True, False, False, False, False, False, False),
    (False, False, False, True, False, False, False, False, False),
    (False, False, False, False, True, False, False, False, False),
    (False, False, False, False, False, True, False, False, False),
    (False, False, False, False, False, False, True, False, False),
    (False, False, False, False, False, False, False, True, False),
    (False, False, False, False, False, False, False, False, True),
]

setup = []
for days in [0, 1, 2, 3, 5, 7, 10, 14]:
    for varsom in [False, True]:
        for regobs in [[]] + list(map(lambda x: [x], regobs_types)):
            for variation in secondary_variations:
                noregions, nocause, temp, collapse, adam, fmt1, fmt4, levels, avy_idx = variation
                if nocause and days < 1:
                    continue
                if (regobs or avy_idx) and days < 2:
                    continue
                if collapse and days <= 2:
                    continue
                setup.append((days, varsom, regobs) + variation)

fds = {}
def get_data(seasons, days, varsom, regobs, noregions, nocause, temp, collapse, adam, fmt1, fmt4, levels, avy_idx):
    if regobs and not avy_idx:
        regobs_avyidx = regobs
    elif regobs and avy_idx:
        regobs_avyidx = regobs + ["AvalancheIndex"]
    elif avy_idx:
        regobs_avyidx = regobs + ["AvalancheIndex"]
    else:
        regobs_avyidx = regobs
    try:
        print("Reading csv")
        labeled_data = LabeledData.from_csv(seasons=seasons, days=days, regobs_types=regobs_avyidx, with_varsom=varsom)
    except CsvMissingError:
        print("Csv missing. Fetching online data. (This takes a long time.)")
        if tuple(regobs_avyidx) not in fds:
            fds[tuple(regobs_avyidx)] = ForecastDataset(regobs_types=regobs_avyidx, seasons=seasons)
        labeled_data = fds[tuple(regobs_avyidx)].label(days=days, with_varsom=varsom)
        labeled_data.to_csv()

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

    return labeled_data, data_id
