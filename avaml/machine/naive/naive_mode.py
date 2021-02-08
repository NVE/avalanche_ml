import dill
import numpy as np
import pandas as pd

from avaml import setenvironment as se
from avaml.aggregatedata.__init__ import DatasetMissingLabel
from avaml.machine import BulletinMachine, AlreadyFittedError, DILL_VERSION, NotFittedError

__author__ = 'arwi'

class NaiveMode(BulletinMachine):
    def __init__(self):
        self.mode = None
        self.multimode = None
        self.mean = None
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

        label = labeled_data.label
        self.primarymode = label["CLASS", ""].mode().iloc[0]
        not_prim = label["CLASS"].columns.get_level_values(0) != ""
        nan = {"": np.nan, "0": np.nan, "00000000": np.nan}
        self.mode = label["CLASS"].loc[:, not_prim].replace(nan).mode().iloc[0]
        self.multimode = label["MULTI"].apply(lambda column: "".join(
            pd.DataFrame(
                column.replace(nan).dropna().apply(lambda x: str(int(x)).zfill(8)).apply(list).to_list()
            ).mode().iloc[0].to_list()
        ))
        self.mean = label["REAL"].mean(axis=0)
        self.y_columns = label.columns

    def predict(self, labeled_data, force_subprobs=False):
        """Predict data using supplied LabeledData.

        :param labeled_data: LabeledData. Dataset to predict. May have empty LabeledData.label.
        :return:             LabeledData. A copy of data, with LabeledData.pred filled in.
        """
        if not self.fitted:
            raise NotFittedError()

        idx = labeled_data.data.index
        out = pd.concat([self.primarymode, self.mode, self.multimode, self.mean])
        out = pd.DataFrame(np.repeat([out.values], len(idx), axis=0), columns=self.y_columns, index=idx)

        ld = labeled_data.copy()
        ld.pred = out
        return ld.valid_pred()

    def dump(self, identifier):
        file_name = f'{se.local_storage}model_nav-mode_v{DILL_VERSION}_{identifier}.dill'

        with open(file_name, 'wb') as handle:
            dill.dump(self, handle)

    @staticmethod
    def load(identifier):
        file_name = f'{se.local_storage}model_nav-mode_v{DILL_VERSION}_{identifier}.dill'
        with open(file_name, 'rb') as handle:
            bm = dill.load(handle)
        return bm
