import re

from avaml.aggregatedata.__init__ import ForecastDataset, LabeledData, REG_ENG, CsvMissingError
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import MultiTaskElasticNet

from avaml.machine.sk_classifier import SKClassifierMachine

train_seasons = ["2018-19", "2019-20"]
test_seasons = ["2017-18"]

prim_class_weight = {
#    "danger_level": {'4': 2, '1': 2},
}

class_weight = {
#    "cause": {'new-snow': 2},
}

def classifier_creator(indata, outdata, class_weight=None):
    return DecisionTreeClassifier(max_depth=7, class_weight=class_weight)


def regressor_creator(indata, outdata):
    return MultiTaskElasticNet(max_iter=3000)


model_prefix = ''
days = 7
regobs_types = [
    "Faretegn",
    "Tester",
    "Skredaktivitet",
    "Snødekke",
    "Skredproblem",
    "Skredfarevurdering"
]

try:
    print("Reading training csv")
    training_data = LabeledData.from_csv(seasons=train_seasons, days=days, regobs_types=regobs_types, with_varsom=True)
except CsvMissingError:
    print("Csv missing. Fetching online data. (This takes a long time.)")
    training_data = ForecastDataset(seasons=train_seasons, regobs_types=regobs_types).label(days=days, with_varsom=True)
    training_data.to_csv()

try:
    print("Reading testing csv")
    testing_data = LabeledData.from_csv(seasons=test_seasons, days=days, regobs_types=regobs_types, with_varsom=True)
except CsvMissingError:
    print("Csv missing. Fetching online data. (This takes a long time.)")
    testing_data = ForecastDataset(seasons=test_seasons, regobs_types=regobs_types).label(days=days, with_varsom=True)
    testing_data.to_csv()

print("Removing 'cause' columns")
training_data.data = training_data.data.loc[:, [not re.search(r"cause", col) for col in training_data.data.columns.get_level_values(0)]]
testing_data.data = testing_data.data.loc[:, [not re.search(r"cause", col) for col in testing_data.data.columns.get_level_values(0)]]
print("Collapsing timeseries")
training_data = training_data.to_time_parameters(orig_days=1)
testing_data = testing_data.to_time_parameters(orig_days=1)
print("Normalizing")
training_data = training_data.normalize()
testing_data = testing_data.normalize(by=training_data)

f1 = None
importances = None
print(f"Training seasons: {train_seasons}")
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

print(f"Testing seasons: {test_seasons}")
predicted_data = ubm.predict(testing_data)
importances = ubm.feature_importances()
f1 = predicted_data.f1()

print("Writing predictions")
predicted_data.pred.to_csv("output/{0}_sk-classifier_pred.csv".format(model_prefix), sep=';')
print("Writing importances")
importances.to_csv("output/{0}_sk-classifier_importances.csv".format(model_prefix), sep=';')
print("Writing F1 scores")
f1.to_csv("output/{0}_sk-classifier_f1.csv".format(model_prefix), sep=";")
