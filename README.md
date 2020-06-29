# avalanche_ml
Applying machine learning in the Norwegian Avalanche warning Service

## Cloning
The repo uses `NVE/varsomdata` as a submodule. To clone it, use

    git clone --recurse-submodules git@github.com:NVE/avalanche_ml.git
or

    git clone --recurse-submodules https://github.com/NVE/avalanche_ml.git

## Creating datasets
`aggregatedata.py` includes the objects `ForecastDataset` and `LabeledData`.

### `ForecastDataset`
This object fetches data from Varsom and RegObs and inserts it into a system of dictionaries
to facilitate the creation of `LabeledData`. The data is cached, as the response times of the API
is very slow.

#### Constructor: `ForecastDataset(regobs_types: Iterable<Int>, seasons: Iterable<String>)`
Raises `RegObsRegTypeError` if `seasons` is specified incorrectly (if the strings are not a
subset of the available observations that can be fetched from RegObs).

* `regobs_types`: What to download from RegObs. E.g., if danger signs and snowpack test are
  wanted, use `("Faretegn", "Tester", "Skredaktivitet")`.
* `seasons`: The seasons to download and use, e.g. `('2017-18', '2018-19', '2019-20')`.

#### Methods
##### `ForecastDataset.label(self, days: Int) -> LabeledData`
* `days`: How far back in time values should data be included. 
  The number of days actually used for each modality is not obvious.
  The reason for this is to make sure that each kind of data contain
  the same number of data points, if we want to use some time series
  frameworks that are picky about such things.
  * If `0`, only weather data for the forecast day is evaluated.
  * If `1`, day 0 is used for weather, 1 for Varsom.
  * If `2`, day 0 is used for weather, 1 for Varsom.
  * If `3`, days 0-1 is used for weather, 1-2 for Varsom, 2-3 for RegObs.
  * If `5`, days 0-3 is used for weather, 1-4 for Varsom, 2-5 for RegObs.

### `LabeledData`
It contains two `DataFrame`s, the features and the labels, and supports som operations on those.

#### Constructor: `LabeledData(data: DataFrame, label: DataFrame, days: Int, regobs_types: Iterable<String>)`
* `data`: A `DataFrame` containing features. Probably created by `ForecastDataset.label()`.
* `label`: A `DataFrame` containing labels. Probably created by `ForecastDataset.label()`.
* `days`: See `ForecastDataset.label()`.
* `regobs_types`: See `ForecastDataset()`.

#### Methods
##### `LabeledData.normalize(self) -> LabeledData`
Scale the data to the range [0, 1].

##### `LabeledData.denormalize(self) -> LabeledData`
Invert `LabeledData.normalize()`.

##### `LabeledData.to_timeseries(self) -> (numpy.ndarray, list<Int>)`
Converts `LabeledData.data` to a `numpy.ndarray` of shape `(rows, time_steps, modalities)` and a list of
the names of the features.

##### `LabeledData.to_csv(self)`'
Writes a csv-file in `varsomdata/localstorage` named according to the properties of the dataset.
A `label.csv` is also always written.

##### `LabeledData.copy(self) -> LabeledData`
Makes a copy of the object.

#### Static methods
##### `LabeledData.from_csv(days: Int, regobs_types: Iterable<String>) -> LabeledData`
Reads a csv-file from localstorage with the given properties. Raises a `CsvMissingError` if
a csv-file with the given properties is not found in localstorage.

* `days`: See `ForecastDataset.label()`.
* `regobs_types`: See `ForecastDataset()`.

#### Attributes
* `LabeledData.data`: A DataFrame of the features.
* `LabeledData.label`: A DataFrame of the labels.

### Example program
```python
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

print("Fetching data (this will take a very long time the first run, then it is cached)")
forecast_dataset = ForecastDataset(regobs_types=("Faretegn", "Tester", "Skredaktivitet"))
print("Labeling data")
labeled_data = forecast_dataset.label(days=3)
print("Writing .csv")  # The .csv is always written using denormalized data.
labeled_data.to_csv()
print("Normalizing data")
labeled_data = labeled_data.normalize()
print("Transforming label")
le = LabelEncoder()
labels = labeled_data.label.values.ravel()
le.fit(labels)
labels = le.transform(labels)
print("Running classifier")
kf = KFold(n_splits=5, random_state=1, shuffle=True)
f1 = []
for split_idx, (train_index, test_index) in enumerate(kf.split(labeled_data.data)):
    X_train = labeled_data.data.iloc[train_index]
    X_test = labeled_data.data.iloc[test_index]
    y_train = labels[train_index]
    y_test = labels[test_index]
    clf = SVC()
    clf.fit(X_train, y_train)
    predictions = le.inverse_transform(clf.predict(X_test))
    danger = le.inverse_transform(y_test)
    f1_part = f1_score(danger, predictions, labels=le.classes_, average='weighted')
    f1.append(f1_part)
    print(f"F1 split {split_idx}: {f1_part}")
print(f"F1 mean: {sum(f1) / len(f1)}")
```