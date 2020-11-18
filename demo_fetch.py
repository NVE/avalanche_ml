import sys
from sklearn.cluster import AgglomerativeClustering

from avaml.aggregatedata import ForecastDataset, LabeledData, CsvMissingError, DatasetMissingLabel, \
    NoBulletinWithinRangeError, NoDataFoundError
from sklearn.tree import DecisionTreeClassifier

from avaml.machine.sk_clustered import SKClusteringMachine
from datetime import date

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
expected_errors = (NoBulletinWithinRangeError, DatasetMissingLabel, NoDataFoundError)

tag_label = sys.argv[1] if len(sys.argv) > 1 else ""
tag_nolabel = tag_label + "-nolabel"
today = date.today()
#today = date.fromisoformat("2020-03-11")

getData = lambda regobs, days_, with_varsom: ForecastDataset\
    .date(regobs_types=regobs, date=today, days=days_, use_label=use_label)\
    .label(days=days_, with_varsom=with_varsom)\
    .to_csv(tag=tag)

for use_label, label_str, tag in [(False, "without labels", tag_nolabel), (True, "with labels", tag_label)]:
    print(f"Saving dataset for {today.isoformat()}, 0 days, {label_str}.")
    try:
        getData([], 0, False)
    except expected_errors:
        print(f"Found no data for {today.isoformat()}, 0 days, {label_str}.")
        break

    print(f"Saving dataset for {today.isoformat()}, 1 days, without varsom, {label_str}.")
    try:
        getData([], 1, False)
    except expected_errors:
        print(f"Found no data for {today.isoformat()}, 1 days, without varsom, {label_str}.")
        break

    print(f"Saving dataset for {today.isoformat()}, 1 days, with varsom, {label_str}.")
    try:
        getData([], 1, True)
    except expected_errors:
        print(f"Found no data for {today.isoformat()}, 1 days, with varsom, {label_str}.")
        except_varsom = True

    except_days = False
    except_varsom = False
    for days in [2, 3, 5, 7, 10, 14]:
        if except_days:
            break
        for varsom, varsom_str in [(False, "without varsom"), (True, "with varsom")]:
            if except_varsom and varsom:
                continue
            try:
                for regobs, regobs_str in [([], "without regobs"), (regobs_types, "with regobs")]:
                    print(f"Saving dataset for {today.isoformat()}, {days} days, {varsom_str}, {regobs_str}, {label_str}.")
                    getData(regobs, days, varsom)
            except expected_errors:
                print(f"Found no data for {today.isoformat()}, {days} days, {varsom_str}, {regobs_str}, {label_str}.")
                if varsom:
                    except_varsom = True
                else:
                    except_days = True
                break
