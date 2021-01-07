import sys
import os

import dill

from avaml.vector.__init__ import Score

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.machine.meta import MetaMachine

id = "meta_test"

meta_model = MetaMachine().load(id)
predictions = meta_model.predict()
score = Score(predictions).calc()
score.to_csv(f"{root}/output/meta_mean_score_2021.csv", sep=";")
predictions.pred.to_csv(f"{root}/output/meta_mean_pred_2021.csv", sep=";")
predictions.label.to_csv(f"{root}/output/meta_mean_label_2021.csv", sep=";")
