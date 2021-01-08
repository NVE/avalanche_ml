from avaml.aggregatedata import ForecastDataset, LabeledData, REG_ENG, CsvMissingError
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import MultiTaskElasticNet
import pandas as pd

from avaml.machine.sk_classifier import SKClassifierMachine

prim_class_weight = {
    "danger_level": {'4': 2, '1': 2},
}

class_weight = {
    "cause": {'new-snow': 2},
}

def classifier_creator(indata, outdata, class_weight=None):
    return DecisionTreeClassifier(max_depth=7, class_weight=class_weight)


def regressor_creator(indata, outdata):
    return MultiTaskElasticNet()


model_prefix = ''
days = 7
regobs_types = list(REG_ENG.keys())
labeled_data = None
try:
    print("Reading csv")
    labeled_data = LabeledData.from_csv(days=days, regobs_types=regobs_types, with_varsom=True)
except CsvMissingError:
    print("Csv missing. Fetching online data. (This takes a long time.)")
    labeled_data = ForecastDataset(regobs_types=regobs_types).label(days=days, with_varsom=True)
    labeled_data.to_csv()

labeled_data = labeled_data.normalize()
#labeled_data.label.drop("REAL", axis=1, level=0, inplace=True)
#labeled_data.pred.drop("REAL", axis=1, level=0, inplace=True)

f1 = None
importances = None
strat = ("CLASS", "", "danger_level")
for split_idx, (training_data, testing_data) in enumerate(labeled_data.kfold(5, stratify=strat)):
    print(f"Training fold: {split_idx}")
    bm = SKClassifierMachine(
        classifier_creator,
        classifier_creator,
        classifier_creator,
        regressor_creator,
        sk_prim_class_weight=prim_class_weight,
        sk_class_weight=class_weight,
    )
    bm.fit(training_data)

    bm.dump(model_prefix)
    ubm = SKClassifierMachine.load(model_prefix)

    print(f"Testing fold: {split_idx}")
    predicted_data = ubm.predict(testing_data)
    labeled_data.pred.loc[predicted_data.pred.index] = predicted_data.pred
    split_imp = ubm.feature_importances()
    importances = split_imp if importances is None else importances + (split_imp - importances) / (split_idx + 1)
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
predicted_data.pred.to_csv("output/{0}_sk-classifier_pred.csv".format(model_prefix), sep=';')
print("Writing importances")
importances.to_csv("output/{0}_sk-classifier_importances.csv".format(model_prefix), sep=';')
print("Writing F1 scores")
f1.to_csv("output/{0}_sk-classifier_f1.csv".format(model_prefix), sep=";")
