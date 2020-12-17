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
    "Skredhendelse",
    "Snødekke",
    "Skredproblem",
    "Skredfarevurdering"
]

setup = [
    (0, False, [], False),
    (0, False, [], True),
    (1, False, [], False),
    (1, False, [], True),
    (1, True, [], False),
    (1, True, [], True),
]
for days in [2, 3, 5, 7, 10, 14]:
    for varsom in [False, True]:
        for regobs in [[], regobs_types]:
            for temp in [False, True]:
                setup.append((days, varsom, regobs, temp))
