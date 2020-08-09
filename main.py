from aggregatedata import ForecastDataset, LabeledData, REG_ENG, CsvMissingError
from machine import BulletinMachine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import MultiTaskElasticNet

prim_class_weight = {
    "danger_level": {'4': {0: 2, 1: 2}},
}

class_weight = {
    "cause": {'new-snow': {0: 2, 1: 2}},
}

def classifier_creator(indata, outdata, class_weight=None):
    return RandomForestClassifier(n_estimators=100, class_weight=class_weight)

def regressor_creator(indata, outdata):
    return MultiTaskElasticNet()

days = 7
regobs_types = list(REG_ENG.keys())
labeled_data = None
try:
    print("Reading csv")
    labeled_data = LabeledData.from_csv(days=days, regobs_types=regobs_types)
except CsvMissingError:
    print("Csv midding. Fetching online data. (This takes a long time.)")
    labeled_data = ForecastDataset(regobs_types=regobs_types).label(days=days)
    labeled_data.to_csv()

f1 = None
importances = None
for split_idx, (training_data, testing_data) in enumerate(labeled_data.kfold(5)):
    print(f"Training fold: {split_idx}")
    bm = BulletinMachine(
        classifier_creator,
        classifier_creator,
        classifier_creator,
        regressor_creator,
        sk_prim_class_weight=prim_class_weight,
        sk_class_weight=class_weight,
    )
    bm.fit(training_data, epochs=80, verbose=1)

    print(f"Testing fold: {split_idx}")
    predicted_data = bm.predict(testing_data)
    labeled_data.pred.loc[predicted_data.pred.index] = predicted_data.pred
    split_imp = bm.feature_importances()
    importances = split_imp if importances is None else importances + (split_imp - importances) / (split_idx + 1)
    f1_series = predicted_data.f1()
    f1 = f1_series if f1 is None else f1 + (f1_series - f1) / (split_idx + 1)
    break

print("Writing predictions")
predicted_data.pred.to_csv("pred.csv", sep=';')
print("Writing importances")
importances.to_csv("importances.csv", sep=';')
print("Writing F1 scores")
f1.to_csv("f1.csv", sep=";")
