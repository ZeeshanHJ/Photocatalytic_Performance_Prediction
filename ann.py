
import os
import site
site.addsitedir("D:\\AI4Water")



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from SeqMetrics import RegressionMetrics

from ai4water import Model
from ai4water.models import MLP, LSTM
from ai4water.utils.utils import dateandtime_now

from utils import PrepareData


# data preparation

inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

target = "Efficiency"  # Efficiency #k
transformation = "ohe"  # le #ohe
lookback = 0
run_tye="dry_run"  # optimize # dry_run
num_iterations=100

exp_name = f"{target}_{transformation}_{lookback}_{run_tye}_{dateandtime_now()}"

prepare_data = PrepareData()

x,y = prepare_data(inputs, target=target, transformation=transformation,
                          lookback=lookback)

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)

model_config = LSTM(
    units=32,
    num_layers=2,
    activation=None,  # "tanh", "sigmoid", "relu", "leaky_relu", "elu"
    dropout=None,  # 0.0, 0.5
    input_shape=(lookback, train_x.shape[-1],))



# model = Model(
#     model=model_config,
#     output_features=[target],
#     input_features=inputs,
#     lr=0.001,
#     batch_size=32,
#     epochs=500
# )
#
# model.fit(
#     x=train_x, y=train_y.reshape(-1,1),
#     validation_data=(val_x, val_y.reshape(-1,1))
# )

#t,p = model.predict(x=val_x, y=val_y.reshape(-1,1), return_true=True)


# ****************** from config *********************************
cpath = r'D:\Zeeshan\modeling\results\Efficiency_ohe_0_hpo_ann_20220518_135105\1_20220518_151507'
model = Model.from_config_file(os.path.join(cpath, "config.json"))

model.verbosity = 1
w_path = os.path.join(model.path, "weights", "weights_923_1.89222.hdf5")
model.update_weights(w_path)

total_true, total_pred = model.predict(
    x=x, y=y, return_true=True,
    process_results=False,
    plots=["residual", "regression", "prediction", "murphy"],
)

print(RegressionMetrics(total_true, total_pred).r2())

total_data = np.hstack([x, total_true, total_pred.reshape(-1,1)])
total_io = pd.DataFrame(total_data,
                        columns=prepare_data.inputs + [f"true_{target}", f"pred_{target}"])
total_io.to_csv(os.path.join(model.path, "total_data.csv"))