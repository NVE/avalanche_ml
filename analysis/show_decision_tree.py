from sklearn.tree import export_graphviz
from machine import BulletinMachine
import os

bm = BulletinMachine.load('dt')
ds_dt = bm.machines_class['drift-slab']

# TODO: add feature names and classes
export_graphviz(ds_dt, out_file='drift_slab_dt.dot', max_depth=2)

# run "dot -Tpdf drift_slab_dt.dot -o drift_slab_dt.pdf"
os.system("dot -Tpdf drift_slab_dt.dot -o drift_slab_dt.pdf")
