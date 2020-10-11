__author__ = 'arwi'

from avaml import Error


class BulletinMachine:
    def __init__(self):
        """Facilitates training and prediction of avalanche warnings.
        """
        self.y_columns = None
        self.X_columns = None
        self.dummies_columns = None
        self.dtypes = None
        self.row_weight = None
        self.fitted = False

    def fit(self, labeled_data):
        """Fits models to the supplied LabeledData.

        :param labeled_data: LabeledData: Dataset that the models should be fit after.
        """
        raise NotImplementedError("Method not implemented!")

    def predict(self, labeled_data):
        """Predict data using supplied LabeledData.

        :param labeled_data: LabeledData. Dataset to predict. May have empty LabeledData.label.
        :return:             LabeledData. A copy of data, with LabeledData.pred filled in.
        """
        raise NotImplementedError("Method not implemented!")

    def feature_importances(self):
        """Used to get all feature importances of internal classifiers.
        Supplied models must support model.feature_importances_, otherwise they are ignored.

        :return: DataFrame. Feature importances of internal classifiers.
        """
        raise FeatureImportanceMissingError("Machine does not use feature importances!")

    def dump(self, identifier):
        raise NotImplementedError("Method not implemented!")

    @staticmethod
    def load(identifier):
        raise NotImplementedError("Method not implemented!")


class NotImplementedError(Error):
    pass


class AlreadyFittedError(Error):
    pass


class NotFittedError(Error):
    pass


class FeatureImportanceMissingError(Error):
    pass

