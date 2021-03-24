from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.tree import DecisionTreeClassifier

import sys
import os
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
