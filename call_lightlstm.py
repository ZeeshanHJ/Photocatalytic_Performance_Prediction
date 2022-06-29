
import os
import site
site.addsitedir("D:\\AI4Water")

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import pandas as pd
from easy_mpl import plot

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model as KModel

from SeqMetrics import RegressionMetrics
from sklearn.model_selection import train_test_split
from ai4water.utils.utils import dateandtime_now
from ai4water.functional import Model

from utils import PrepareData
from lightlstm import LightLSTM

# data preparation

inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

target = "Efficiency"  # Efficiency #k
transformation = "ohe"  # le # ohe
lookback = 10
return_sequences = True

exp_name = f"{target}_{transformation}_{lookback}_{dateandtime_now()}"

prepare_data = PrepareData()
x,y = prepare_data(inputs,
                   target=target,
                   transformation=transformation,
                   lookback=lookback,
                   return_sequences=return_sequences
                   )

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)
train_x_static, train_x_dyn = train_x[..., 0:-1], train_x[..., -1]
val_x_static, val_x_dyn = val_x[..., 0:-1], val_x[..., -1]
train_x_dyn = np.expand_dims(train_x_dyn, axis=-1)
val_x_dyn = np.expand_dims(val_x_dyn, axis=-1)

num_dyn_inputs= 1
num_static_inputs=28
units=100
batch_size = 4


class FModel(Model):

    def add_layers(self, *args, **kwargs):
        config = args[0]
        inp_dyn = Input(batch_shape=(batch_size, lookback, num_dyn_inputs))
        inp_static = Input(batch_shape=(batch_size, lookback, num_static_inputs))

        lstm = LightLSTM(config['units'],
                         num_static_inputs,
                         return_sequences=return_sequences,
                         activation=config['activation'])(inp_dyn, inp_static)

        out = Dense(1)(lstm)

        if return_sequences:
            out = Flatten()(out)

        return [inp_dyn, inp_static], out



# model_config = {"layers":
#                     {"Input_0": {"batch_shape": (batch_size, lookback, num_dyn_inputs)},
#                 "Input_1": {"batch_shape": (batch_size, lookback, num_static_inputs)}}
#                 }
# _model = FModel(
#     model={"layers": {"units": units, "activation": "relu"}},
#     lr=0.0010053617482820901,
#     patience=800,
#     epochs=1000,
#     verbosity=0,
#     #prefix=prefix or PREFIX,
# )
#
# model.fit(
#     x=[train_x_dyn[0:92], train_x_static[0:92]], y=train_y[0:92].reshape(92, -1),
#     validation_data=([val_x_dyn[0:40], val_x_static[0:40]], val_y[0:40].reshape(40, -1)),
#     batch_size=batch_size,
#     epochs=500,
#
# )
#

# print(RegressionMetrics(val_y[0:40].reshape(-1,1), p.reshape(-1,)).r2())


# ****************** from config *********************************
cpath = r'D:\Zeeshan\modeling\results\Efficiency_ohe_10_hpo_20220519_152136\1_20220519_162029'
_model = FModel.from_config_file(os.path.join(cpath, "config.json"))

_model.verbosity = 1
w_path = os.path.join(cpath, "weights", "weights_999_10.48209.hdf5")
_model.update_weights(w_path)


# ** validation data **
# val_pred = _model.predict([val_x_dyn[0:40], val_x_static[0:40]], y=val_y[0:40],
#                   process_results=False)
#
# val_pred1 = val_pred.reshape(-1, 1)
# val_true1 = val_y[0:40].reshape(-1, 1)
#
# val_data = np.hstack([val_x[0:40].reshape(-1, 29),
#                         val_pred1,
#                         val_true1
#                         ])
#
# val_data = pd.DataFrame(val_data,
#                           columns=prepare_data.inputs + [f"pred_{target}", f"true_{target}"])
# cat_columns = [col for col in val_data.columns if 'catalyst_' in col]
# mapper = {k:v for k,v in zip(cat_columns, prepare_data.cat_encoder.categories_[0])}
# val_data = val_data.rename(columns=mapper)
# # decode name of anions
# anion_columns = [col for col in val_data.columns if 'Anions_' in col]
# mapper = {k:v for k,v in zip(anion_columns, prepare_data.anion_encoder.categories_[0])}
# val_data = val_data.rename(columns=mapper)
# val_data.to_csv(os.path.join(_model.path, "val_data.csv"))


# ** training data **
train_pred = _model.predict([train_x_dyn[0:92], train_x_static[0:92]], y=train_y[0:92],
                  process_results=False)

train_pred1 = train_pred.reshape(-1, 1)
train_true1 = train_y[0:92].reshape(-1, 1)

train_data = np.hstack([train_x[0:92].reshape(-1, 29),
                        train_pred1,
                        train_true1
                        ])

train_data = pd.DataFrame(train_data,
                          columns=prepare_data.inputs + [f"pred_{target}", f"true_{target}"])
cat_columns = [col for col in train_data.columns if 'catalyst_' in col]
mapper = {k:v for k,v in zip(cat_columns, prepare_data.cat_encoder.categories_[0])}
train_data = train_data.rename(columns=mapper)
# decode name of anions
anion_columns = [col for col in train_data.columns if 'Anions_' in col]
mapper = {k:v for k,v in zip(anion_columns, prepare_data.anion_encoder.categories_[0])}
train_data = train_data.rename(columns=mapper)
train_data.to_csv(os.path.join(_model.path, "train_data.csv"))



# ** total data **
# new_samples = 1
# dummy_x = np.zeros((new_samples, 10, 29))
# dummy_y = np.zeros((new_samples, 10))
# new_x = np.vstack([x, dummy_x])
# new_y = np.vstack([y, dummy_y])
#
# xx = [np.expand_dims(new_x[..., -1], axis=-1), new_x[..., 0:-1]]
# total_pred = _model.predict(
#     x=xx,
#     y=new_y,
#     process_results=False,
# )
#
# total_pred1 = total_pred[0:-1].reshape(-1, 1)
# total_true1 = new_y[0:-1].reshape(-1, 1)
# print(RegressionMetrics(total_true1, total_pred1).r2())

