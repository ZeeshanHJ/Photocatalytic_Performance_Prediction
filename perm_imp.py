# permutation importance

import os

import site
site.addsitedir("D:\\AI4Water")

import numpy as np
import pandas as pd

from ai4water.postprocessing import PermutationImportance

from utils import PrepareData, MyModel, decode_with_ohe



inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

target = "Efficiency"
lookback = 0


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




pimp = PermutationImportance(func, X,
                             y,
                             feature_names=inputs.copy(),
                             n_repeats=100,
                             scoring="r2",

                             )

fig = pimp.plot_1d_pimp()