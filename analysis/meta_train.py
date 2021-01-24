import sys
import os

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.machine.meta import MetaMachine

id = "meta_mean_train-2019-20_test-2018-19"

print("Initializing")
meta_model = MetaMachine()
print("Fitting")
meta_model.fit()
print("Dumping")
meta_model.dump(id)
