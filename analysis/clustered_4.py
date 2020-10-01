from sklearn.tree import DecisionTreeClassifier
from aggregatedata import ForecastDataset, LabeledData, REG_ENG, CsvMissingError
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import confusion_matrix
import numpy as np
from machine import BulletinMachine

days = 2
regobs_types = list(REG_ENG.keys())

try:
    print("Reading csv")
    labeled_data = LabeledData.from_csv(days=days, regobs_types=regobs_types, with_varsom=True)
except CsvMissingError:
    print("Csv missing. Fetching online data. (This takes a long time.)")
    labeled_data = ForecastDataset(regobs_types=regobs_types).label(days=days, with_varsom=True)
    labeled_data.to_csv()

print("Normalizing data")
labeled_data = labeled_data.normalize()

print("Dropping irrelevant targets")
labeled_data.label.drop("REAL", axis=1, level=0, inplace=True)
labeled_data.pred.drop("REAL", axis=1, level=0, inplace=True)
labeled_data.label = labeled_data.label.iloc[:, :1]
labeled_data.pred = labeled_data.pred.iloc[:, :1]

def classifier_creator(indata, outdata, class_weight=None):
    return DecisionTreeClassifier(max_depth=7)

f1 = None
importances = None
strat = ("CLASS", "", "danger_level")
for split_idx, (training_data, testing_data) in enumerate(labeled_data.kfold(5, stratify=strat)):
    features = training_data.data
    # Remove region columns from cluster features
    cols = [c for c in features.columns.get_level_values(0).unique() if c[:6] != 'region']
    cluster_features = features[cols]
    danger_level = training_data.label["CLASS", "", "danger_level"].values.astype(np.int)

    # Size 6 was found best empirically
    clusters = 6
    clustering = AgglomerativeClustering(n_clusters=clusters)
    print("Clustering features")
    cluster_labels = clustering.fit_predict(cluster_features)

    confusion = confusion_matrix(danger_level, cluster_labels + 1)[:4, :clusters]
    fours_in_cluster = confusion[3, :] / confusion.sum(axis=0)

    idx = cluster_labels == fours_in_cluster.argmax()
    concentrated_data = training_data.copy()
    concentrated_data.data = concentrated_data.data.loc[idx, :]
    concentrated_data.label = concentrated_data.label.loc[idx, :]
    concentrated_data.pred = concentrated_data.pred.loc[idx, :]
    concentrated_data.row_weight = concentrated_data.row_weight.loc[idx, :]

    print(f"Training fold: {split_idx}")
    cluster_machine = BulletinMachine(
        classifier_creator,
        classifier_creator,
        classifier_creator,
        lambda in_size, out_size: np.zeros(out_size),
    )
    cluster_machine.fit(concentrated_data)

    full_machine = BulletinMachine(
        classifier_creator,
        classifier_creator,
        classifier_creator,
        lambda in_size, out_size: np.zeros(out_size),
    )
    full_machine.fit(training_data)

    print(f"Testing fold: {split_idx}")
    concentrated_pred = cluster_machine.predict(testing_data)
    full_pred = full_machine.predict(testing_data)

    is4_label = testing_data.label['CLASS', '', 'danger_level'] == '4'
    is4_concentrated = concentrated_pred.pred["CLASS", "", "danger_level"] == "4"
    is4_full = full_pred.pred["CLASS", "", "danger_level"] == "4"
    is4_both = np.logical_and(is4_concentrated, is4_full)
    is4_any = np.logical_or(is4_concentrated, is4_full)
    is4_just_conc = np.logical_and(is4_concentrated, np.logical_not(is4_both))
    print(f"Found {is4_full.sum()} 4s in ordinary machine.")
    print(f"Found {is4_concentrated.sum()} 4s in clustered machine.")
    print(f"Found {is4_just_conc.sum()} 4s in clustered machine that did not exist in ordinary machine.")
    print(f"Found {is4_any.sum()} 4s in total (out of {is4_full.shape[0]} predictions).")
    print(f"In reality, there are {is4_label.sum()} 4s in this fold.")

    print("Merging cluster 4s into mainline predictions")
    full_pred.pred.loc[is4_concentrated, :] = concentrated_pred.pred.loc[is4_concentrated, :]

    labeled_data.pred.loc[full_pred.pred.index] = full_pred.pred
    split_imp = full_machine.feature_importances()
    f1_series = full_pred.f1()
    f1 = f1_series if f1 is None else f1 + (f1_series - f1) / (split_idx + 1)
    #break


print("Writing predictions")
labeled_data.pred.to_csv("../clustered_4_pred.csv", sep=';')
print("Writing F1 scores\n")
f1.to_csv("../clustered_4_f1.csv", sep=";")
