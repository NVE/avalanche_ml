import pandas as pd
import matplotlib.pylab as plt
from pathlib import Path


# Set filepath
root = Path(r'~/PycharmProjects/avalanche_ml/').expanduser()
report_dir = root / 'reports'
f1_file = root / 'output/_sk-classifier_f1.csv'
pred_file = root / 'output/_sk-classifier_pred.csv'
importances_file = root / 'output/_sk-classifier_importances.csv'
prefix = f1_file.stem


def rm_bad_char(s: str):
    # initializing bad_chars_list
    bad_chars = [';', ':', '!', "*", "?", "."]
    if any(bc in s for bc in bad_chars):
        for bc in bad_chars:
            s = s.replace(bc, '_')
        return s
    else:
        return s


# Read F1 score file
f1_df = pd.read_csv(f1_file, sep=';', header=[0], index_col=[0, 1, 2, 3])
f1_df.head()

_f1_df = f1_df.drop(['REAL', 'MULTI']).sort_values(by='f1', axis='index')
_ax1 = _f1_df[['f1', 'precision', 'recall']].plot(kind='barh', stacked=True, figsize=(15, 30))
_ax1.set(xlabel='Scores',
         ylabel='Attribute',
         title='Scores for {0}'.format(prefix))
plt.yticks(rotation=30)
plt.tight_layout()
plt.savefig(report_dir / '{0}_scores.pdf'.format(prefix))


# Read importance file
importances_df = pd.read_csv(importances_file, sep=';', header=[0, 1], index_col=[0, 1])
importances_df.head()

for c in importances_df.columns:
    _df = importances_df.nlargest(15, c, keep='all')
    _ax2 = _df[c].sort_values().plot(kind='barh', figsize=(15, 10))
    _ax2.set(xlabel='Feature',
             ylabel='Importance',
             title='Importance for {0}, {1}'.format(c[0], c[1]))
    plt.tight_layout()
    plt.savefig(report_dir / '{2}_importance_{0}_{1}.pdf'.format(c[0], rm_bad_char(c[1]), prefix))
