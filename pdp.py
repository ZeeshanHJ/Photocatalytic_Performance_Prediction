# Partial Dependence Plot

import os

import site
site.addsitedir("D:\\AI4Water")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ai4water.postprocessing import PartialDependencePlot

from utils import PrepareData, MyModel, decode_with_ohe



inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

target = "Efficiency"
lookback = 0

# prediction_value_change
prepare_data = PrepareData()

X, y = prepare_data(inputs.copy(),
                   target=target,
                   transformation="le",
                   lookback=lookback,
                   )

c_path = r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\CatBoostRegressor\best\20220512_115620"
model = MyModel.from_config_file(os.path.join(c_path, "config.json"))

model.verbosity = 1
model.update_weights()


def func(x:np.ndarray):

    cat_enc = x[:, 0].reshape(-1, 1).astype(np.int16)
    cat_dec = prepare_data.cat_encoder.inverse_transform(cat_enc)

    an_enc = x[:, 6].reshape(-1, 1).astype(np.int16)
    an_dec = prepare_data.anion_encoder.inverse_transform(an_enc)

    x_df = pd.DataFrame(x, columns=prepare_data.inputs)

    x_df['Catalyst_type'] = cat_dec
    x_df['Anions'] = an_dec

    x_df_dec, new_inputs, c_encdr, an_encdr = decode_with_ohe(x_df, inputs.copy())

    x = x_df_dec[new_inputs]

    pred = model.predict(x=x)

    return pred


pdp = PartialDependencePlot(func,
                             X,
                            num_points=20,
                             feature_names=inputs.copy(),
                            show=False,
                             )

xticklabels = np.arange(-50, 300, 30)

for feature in ['time']:
    plt.close('all')
    ax = pdp.plot_1d(feature, ice=False)
    ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    #ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels())
    fname = os.path.join(c_path, f"{feature}_no_icepdp.png")
    #plt.savefig(fname, bbox_inches="tight")
    plt.show()

#pdp.plot_interaction(['time', 'Catalyst_loading'],plot_type="surface")