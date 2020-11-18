from sklearn.cluster import AgglomerativeClustering

from avaml.aggregatedata import ForecastDataset, LabeledData, CsvMissingError
from sklearn.tree import DecisionTreeClassifier

from avaml.machine.sk_clustered import SKClusteringMachine
from datetime import date

model_prefix = 'cluster_today_demo'
days = 3
regobs_types = [
    "Faretegn",
    "Tester",
    "Skredaktivitet",
    "Skredhendelse",
    "Snødekke",
    "Skredproblem",
    "Skredfarevurdering"
]
varsom = False

try:
    bm = SKClusteringMachine.load(model_prefix)
except FileNotFoundError:
    try:
        print("Reading csv")
        labeled_data = LabeledData.from_csv(
            seasons=["2017-18", "2018-19", "2019-20"],
            days=days,
            regobs_types=regobs_types,
            with_varsom=varsom,
        )
    except CsvMissingError:
        print("Csv missing. Fetching online data. (This takes a long time.)")
        labeled_data = ForecastDataset(regobs_types=regobs_types).label(days=days, with_varsom=varsom)
        labeled_data.to_csv()

    labeled_data = labeled_data.normalize()

    f1 = None
    importances = None
    strat = ("CLASS", "", "danger_level")

    print(f"Training model")
    dt = DecisionTreeClassifier(max_depth=7, class_weight={"1": 1, "2": 1, "3": 1, "4": 1})
    clustering = AgglomerativeClustering(n_clusters=20)

    bm = SKClusteringMachine(dt, clustering)
    bm.fit(labeled_data)

    bm.dump(model_prefix)

#single_date = date.fromisoformat("2020-03-01")
single_date = date.fromisoformat("2020-11-15")
fd = ForecastDataset.date(regobs_types=regobs_types, date=single_date, days=days, use_label=False)
today_data = fd.label(days=days, with_varsom=varsom)
print(today_data.label)
today_data.to_csv(tag="single-date")

predicted_data = bm.predict(today_data)
print(predicted_data.pred.iloc[:, :5])
