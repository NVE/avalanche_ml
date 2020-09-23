from sklearn.tree import export_graphviz
from machine import BulletinMachine
import os

target = 'drift-slab'
m = 'dt'

bm = BulletinMachine.load(m)
ds_dt = bm.machines_class[target]
fn = list(bm.X_columns.to_flat_index())
# cn = list(bm.y_columns.to_flat_index())
cn = list(bm.dummies["CLASS"][target].columns.to_flat_index())
# TODO: add feature names and classes
export_graphviz(ds_dt, out_file='{0}_{1}.dot'.format(target, m),
                # max_depth=3,
                feature_names=fn, class_names=cn)

# run dot
os.system("dot -Tpdf {0}_{1}.dot -o {0}_{1}.pdf".format(target, m))
