# Making predictions
`machine.py` includes the object `BulletinMachine`. The object of it is to facilitate the multiple
models used to generate a full AvalancheWarning.

##`BulletinMachine`

### Constructor `BulletinMachine(ml_prim_creator: fn(), ml_class_creator: fn(), ml_multi_creator: fn(), ml_real_creator: fn())`
All supplied models must support the methods `model.fit(X, y)` and `model.predict(X)`. It is preferable if they also
support `model.feature_importances_`. If `model.predict()` supports the arguments `epochs` and `verbosity`, those
will be forwarded to the model.

* `ml_prim_creator`: `fn(in_size: Int, out_size: Int) -> classifier`, used to solve primary
problems, such as "danger_level" and "problem_n". Preferably softmax output.
* `ml_class_creator`: `fn(in_size: Int, out_size: Int) -> classifier`, used to solve secondary problems,
such as "cause" or "dist". Preferably softmax output.
* `ml_multi_creator`: `fn(in_size: Int, out_size: Int) -> classifier`, used to solve multilabel problems,
such as "aspect". Must be k-of-n-hot.
* `ml_real_creator`: `fn(in_size: Int, out_size: Int) -> classifier`, used to solve for real numbers. Must
support multiple outputs.

### Methods
#### `BulletinMachine.fit(self, labeled_data: LabeledData, epochs: Int, verbosity: Int)`
* `labeled_data`: Dataset that the models should be fit after.
* `epochs`: Number of epochs to train. Ignored if the supplied model doesn't support the parameter.
* `verbose`: Verbosity of the models. Ignored if not supported of the supplied models.

#### `BulletinMachine.predict(self, labeled_data: LabeledData) -> LabeledData`
* `labeled_data`: Dataset to predict. May have empty `labeled_data.label`.

Returns a copy of `data` with `data.pred` filled in.

#### `BulletinMachine.feature_importances(self) -> pandas.DataFrame`
Used to get all feature importances of internal classifiers. Supplied models must support
`model.feature_importances_`, otherwise they are ignored.

Returns feature importances of internal classifiers.