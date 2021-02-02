import sys
import os

from avaml.score.__init__ import Score

root = f"{os.path.dirname(os.path.abspath(__file__))}/.."
sys.path.insert(0, root)

from avaml.machine.meta import MetaMachine

seasons = ["2017-18", "2018-19", "2019-20"]
f1 = None
for split_idx, without_season in enumerate(seasons):
    id = f"meta_mean_without_{without_season}"
    train_seasons = seasons.copy()
    train_seasons.remove(without_season)

    print("Initializing")
    meta_model = MetaMachine()
    print(f"Fitting with {train_seasons}")
    meta_model.fit(seasons=train_seasons)
    print("Dumping")
    meta_model.dump(id)
    #meta_model = MetaMachine().load(id)
    print(f"Predicting {without_season}")
    predictions = meta_model.predict(seasons=[without_season])

    f1_series = predictions.f1()

    score = Score(predictions).calc()
    score.to_csv(f"{root}/output/meta_mean_score_{without_season}.csv", sep=";")
    predictions.pred.to_csv(f"{root}/output/meta_mean_pred_{without_season}.csv", sep=";")
    predictions.label.to_csv(f"{root}/output/meta_mean_label_{without_season}.csv", sep=";")
    f1_series.to_csv(f"{root}/output/meta_mean_f1_{without_season}.csv", sep=";")
