import sys
import os

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.machine.meta import MetaMachine

id = "meta_test"

meta_model = MetaMachine()
meta_model.fit()
meta_model.dump(id)
