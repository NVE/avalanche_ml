from sklearn.cluster import AgglomerativeClustering
import pandas as pd

from avaml.aggregatedata import ForecastDataset, LabeledData, CsvMissingError
from sklearn.tree import DecisionTreeClassifier

from avaml.machine.sk_clustered import SKClusteringMachine

model_prefix = ''
days = 2
regobs_types = [
    "Faretegn",
    "Tester",
    "Skredaktivitet",
    "Skredhendelse",
    "Snødekke",
    "Skredproblem",
    "Skredfarevurdering"
]

try:
    print("Reading csv")
    labeled_data = LabeledData.from_csv(days=days, regobs_types=regobs_types, with_varsom=True)
except CsvMissingError:
    print("Csv missing. Fetching online data. (This takes a long time.)")
    labeled_data = ForecastDataset(regobs_types=regobs_types).label(days=days, with_varsom=True)
    labeled_data.to_csv()

labeled_data = labeled_data.normalize()

f1 = None
strat = ("CLASS", "", "danger_level")
for split_idx, (training_data, testing_data) in enumerate(labeled_data.kfold(5, stratify=strat)):
    print(f"Training fold: {split_idx}")
    dt = DecisionTreeClassifier(max_depth=7, class_weight={"1": 1, "2": 1, "3": 1, "4": 1})
    clustering = AgglomerativeClustering(n_clusters=20)

    bm = SKClusteringMachine(dt, clustering)
    bm.fit(training_data)

    bm.dump(model_prefix)
    ubm = SKClusteringMachine.load(model_prefix)

    print(f"Testing fold: {split_idx}")
    predicted_data = ubm.predict(testing_data)
    labeled_data.pred.loc[predicted_data.pred.index] = predicted_data.pred

    f1_series = predicted_data.f1()
    idx = f1_series.index if f1 is None else list(set(f1_series.index.to_list()).intersection(set(f1.index.to_list())))
    f1_series = pd.DataFrame(f1_series, index=idx).sort_index()
    f1_series.loc[["CLASS", "MULTI"], ["f1", "precision", "recall"]].fillna(0, inplace=True)
    if f1 is None:
        f1 = f1_series
    else:
        f1 = pd.DataFrame(f1, index=idx).sort_index()
        f1 = f1 + (f1_series - f1) / (split_idx + 1)

print("Writing predictions")
predicted_data.pred.to_csv("output/{0}_sk-cluster_pred.csv".format(model_prefix), sep=';')
print("Writing F1 scores")
f1.to_csv("output/{0}_sk-cluster_f1.csv".format(model_prefix), sep=";")
