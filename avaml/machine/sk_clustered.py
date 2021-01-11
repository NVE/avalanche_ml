import os

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.tree import export_graphviz

from avaml import setenvironment as se, _NONE
from avaml.aggregatedata import DatasetMissingLabel
from avaml.machine import BulletinMachine, AlreadyFittedError, DILL_VERSION

__author__ = 'arwi'

class SKClusteringMachine(BulletinMachine):
    def __init__(
            self,
            danger_classifier,
            clusterer,
    ):
        """Facilitates training and prediction of avalanche warnings. Only supports sklearn models.

        :param danger_classifier:      classifier: Used to predict danger level.
        :param clusterer:              clusterer: Used to group bulletins toghether.
        """
        self.classifier = danger_classifier
        self.clusterer = clusterer
        self.mode = {}
        self.cluster_ids = {}
        self.cluster_features = {}
        self.cols = None
        super().__init__()

    def fit(self, labeled_data):
        """Fits models to the supplied LabeledData.

        :param labeled_data: LabeledData: Dataset that the models should be fit after.
        """
        if labeled_data.label is None:
            raise DatasetMissingLabel()

        if self.fitted:
            raise AlreadyFittedError()
        self.fitted = True

        X = labeled_data.data
        y = labeled_data.label

        self.X_columns = X.columns
        self.y_columns = y.columns
        self.dummies_columns = labeled_data.to_dummies()['label'].columns
        self.dtypes = y.dtypes.to_dict()
        self.row_weight = labeled_data.row_weight

        self.classifier.fit(X, y["CLASS", _NONE, "danger_level"], labeled_data.row_weight)

        # Filter out region columns
        cols = [c for c in X.columns.get_level_values(0).unique() if c[:6] != 'region']
        self.cols = cols

        for dlevel in range(1, 5):
            dlevel_rows = y["CLASS", _NONE, "danger_level"] == str(dlevel)
            # Remove region columns from cluster features
            cluster_features = X.loc[dlevel_rows][cols]
            cluster_labels = y.loc[dlevel_rows]

            cluster_ids = self.clusterer.fit_predict(cluster_features)

            self.cluster_ids[dlevel] = cluster_ids
            self.cluster_features[dlevel] = cluster_features

            #Creating labels from training clusters
            label_mode = pd.DataFrame(0, columns=y.columns, index=set(cluster_ids))

            for c_id in set(cluster_ids):
                c_labels = cluster_labels.loc[cluster_ids == c_id]
                label_mode.loc[c_id, pd.IndexSlice["CLASS", _NONE]] = c_labels["CLASS", ""].mode(dropna=False).iloc[0].values

                # Fixing problem ordering
                problem_1 = label_mode.loc[c_id, ("CLASS", _NONE, "problem_1")]
                problem_2 = label_mode.loc[c_id, ("CLASS", _NONE, "problem_2")]
                problem_3 = ""
                # Evaluate problem two excluding things equal to problem one
                if problem_1 == problem_2:
                    problem_12_neq = c_labels["CLASS", _NONE, "problem_2"] != problem_1
                    if problem_12_neq.sum():
                        problem_2 = c_labels.loc[problem_12_neq, ("CLASS", _NONE, "problem_2")].mode().iloc[0]
                        label_mode.loc[c_id, ("CLASS", _NONE, "problem_2")] = problem_2
                # Problems three is evaluated from problem two
                if problem_2 != "":
                    problem_12_not = c_labels["CLASS", _NONE, "problem_2"] != problem_1
                    problem_22_not = c_labels["CLASS", _NONE, "problem_2"] != problem_2
                    problem_not = np.logical_and(problem_12_not, problem_22_not)
                    if problem_not.sum():
                        problem_3 = c_labels.loc[problem_not, ("CLASS", _NONE, "problem_2")].mode().iloc[0]
                label_mode.loc[c_id, ("CLASS", _NONE, "problem_3")] = problem_3
                label_mode.loc[c_id, ("CLASS", _NONE, "problem_amount")] = len(list(filter(
                    lambda x: x != "",
                    [problem_1, problem_2, problem_3]
                )))

                # Calculate mode for matched subproblem
                problem_cols = c_labels["CLASS", _NONE][["problem_1", "problem_2", "problem_3"]]
                subprobs = set(filter(
                    lambda x: x != _NONE,
                    np.ravel(label_mode.loc[c_id]["CLASS", _NONE][["problem_1", "problem_2", "problem_3"]])
                ))
                for subprob in subprobs:
                    rows = np.any(np.char.equal(problem_cols.values.astype("U"), subprob), axis=1)
                    if np.any(rows):
                        filtered_label_mode = c_labels.loc[rows, pd.IndexSlice[:, subprob]].mode().iloc[0]
                        label_mode.loc[c_id, pd.IndexSlice[:, subprob]] = filtered_label_mode.values

            self.mode[dlevel] = label_mode

    def predict(self, labeled_data, force_subprobs=False):
        """Predict data using supplied LabeledData.

        :param labeled_data: LabeledData. Dataset to predict. May have empty LabeledData.label.
        :return:             LabeledData. A copy of data, with LabeledData.pred filled in.
        """
        y = pd.DataFrame(
            index=labeled_data.data.index,
            columns=self.y_columns
        ).fillna(0).astype(self.dtypes)
        y["CLASS", "", "danger_level"] = self.classifier.predict(labeled_data.data)

        for dlevel in range(1, 5):
            mode = self.mode[dlevel]
            dlevel_rows = y["CLASS", "", "danger_level"] == str(dlevel)
            if dlevel_rows.sum():
                X = labeled_data.data.loc[dlevel_rows, self.cols]

                distances = pairwise_distances(self.cluster_features[dlevel], X)
                c_ids = self.cluster_ids[dlevel][np.argmin(distances, axis=0)]
                y.loc[dlevel_rows] = self.mode[dlevel].values[c_ids]

        ld = labeled_data.copy()
        ld.pred = y
        ld.pred = ld.pred.fillna("").astype("U")
        ld.pred["REAL"] = ld.pred["REAL"].replace("", 0).astype(np.float)
        return ld

    def feature_importances(self):
        """Used to get all feature importances of internal classifiers.
        Supplied models must support model.feature_importances_, otherwise they are ignored.

        :return: DataFrame. Feature importances of internal classifiers.
        """
        importances = {}
        try:
            importances[('CLASS', "danger_level")] = self.classifier.feature_importances_
        except AttributeError:
            pass
        df = pd.DataFrame(importances, index=self.X_columns)
        df.index.set_names(["feature_name", "day"], inplace=True)
        return df

    def dt_pdf(self, file_name):
        export_graphviz(self.classifier, out_file=f'{file_name}.dot',
                        feature_names=self.X_columns.to_flat_index(), class_names=["1", "2", "3", "4"])

        os.system(f"dot -Tpdf {file_name}.dot -o {file_name}.pdf")

    def dump(self, identifier):
        file_name = f'{se.local_storage}model_skclust_v{DILL_VERSION}_{identifier}.dill'

        with open(file_name, 'wb') as handle:
            dill.dump(self, handle)

    @staticmethod
    def load(identifier):
        file_name = f'{se.local_storage}model_skclust_v{DILL_VERSION}_{identifier}.dill'
        with open(file_name, 'rb') as handle:
            bm = dill.load(handle)
        return bm
