import sys
import os

import dill

from avaml.score.__init__ import Score

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.machine.meta import MetaMachine

id = "meta_mean_train-2019-20_test-2018-19"
season = "2017-18"

meta_model = MetaMachine().load(id)
predictions = meta_model.predict(seasons=[season])
score = Score(predictions).calc()
f1 = predictions.f1()
score.to_csv(f"{root}/output/meta_meanmin_score_{season}.csv", sep=";")
predictions.pred.to_csv(f"{root}/output/meta_meanmin_pred_{season}.csv", sep=";")
predictions.label.to_csv(f"{root}/output/meta_meanmin_label_{season}.csv", sep=";")
f1.to_csv(f"{root}/output/meta_meanmin_f1_{season}.csv", sep=";")
