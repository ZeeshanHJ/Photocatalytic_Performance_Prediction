
import os
import random

import site
site.addsitedir("D:\\AI4Water")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai4water.utils.visualizations import plot_edf

paths = {
    "LGBM": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\LGBMRegressor\best\20220512_122739",

    "GaussianProcess": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\GaussianProcessRegressor\best\20220512_120923",

    "GradientBoosting": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\GradientBoostingRegressor\best\20220512_121206",

    #"ExtraTree": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\ExtraTreeRegressor\best\20220512_120437",

    "HistGradientBoosting": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\HistGradientBoostingRegressor\best\20220512_121952",

    "ExtraTrees": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\ExtraTreesRegressor\best\20220512_120748",

    "XGBoost": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\XGBRegressor\best\20220512_132337",

    "CatBoost": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\CatBoostRegressor\best\20220512_115620",

    "DecisionTree": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\DecisionTreeRegressor\best\20220512_115801",

    "KNeighbors": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\KNeighborsRegressor\best\20220512_122350",

    "Bagging": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\BaggingRegressor\best\20220512_114637",

    #"MLPRegressor": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\MLPRegressor\best\20220512_124521",

    "ANN": r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_hpo_ann_20220518_135105\1_20220518_151507",

    "LightLSTM": r"D:\Zeeshan\modeling\results\Efficiency_ohe_10_hpo_20220519_152136\1_20220519_162029",
}


max_val = 40
markers = ['.', '-.', '--.', 'v']


def get_cmap(cm: str, num_cols: int, low=0.0, high=1.0):

    cols = getattr(plt.cm, cm)(np.linspace(low, high, num_cols))
    return cols

dir_name = os.path.join(os.getcwd(), "edfs")
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

for cmap in ["tab20"]:# # plt.colormaps():

    colors = get_cmap("tab20", len(paths), low=0.1)

    fig, ax = plt.subplots()
    for idx, (mname, path) in enumerate(paths.items()):

        fpath = os.path.join(path, "total_data.csv")
        df = pd.read_csv(fpath)

        error = np.abs(df['true_Efficiency'].values - df['pred_Efficiency'].values)
        error = np.where(error>max_val, max_val, error)

        plot_edf(error,
                 marker=random.choice(markers),
                 color=colors[idx],
                 ax=ax,
                 xlabel="Absote Error",
                 label=mname)

    plt.legend()
    #plt.savefig(os.path.join(dir_name, f"{cmap}_{max_val}"))
    #plt.title(cmap)
    #plt.show()