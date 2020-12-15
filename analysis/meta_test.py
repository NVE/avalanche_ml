import sys
import os

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.machine.meta import MetaMachine

id = "meta_test"

meta_model = MetaMachine().load(id)
predictions = meta_model.predict(csv_tag="test_0944")
predictions.pred.to_csv(f"{root}/output/meta_mean_new_pred_2021.csv", sep=";")
predictions.label.to_csv(f"{root}/output/meta_mean_new_label_2021.csv", sep=";")
