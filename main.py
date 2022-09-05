"""
This file shows application of simple LSTM for efficiency prediction
"""
import site
site.addsitedir("D:\\mytools\\AI4Water")

import ai4water
print(ai4water.__version__)

import numpy as np
np.set_printoptions(suppress=True, linewidth=150)

from sklearn.model_selection import train_test_split
from utils import PrepareData

from ai4water import Model

inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

target = "Efficiency" # Efficiency # k
transformation = "ohe"# le #ohe
lookback = 10

prepare_data = PrepareData()
x,y = prepare_data(inputs, target=target, transformation=transformation,
                          lookback=lookback)

train_x, val_x, train_y, val_y = train_test_split(x,y, test_size=0.3, random_state=313)

# # model building
layers = {
    "Input": {"shape": (lookback, len(prepare_data.inputs))},
    "LSTM": {"units": 32, "return_sequences": True},
    "LSTM_1": {"units": 1, "return_sequences": True},
    "Flatten": {},
    "Dense": 1
}

model = Model(
    model={"layers": layers},
    train_fraction=1.0,
    val_fraction=0.3,
    split_random=True,
    input_features=prepare_data.inputs,
    output_features=['Efficiency'],
    x_transformation="minmax",
    epochs=500
)


# # model Training
h = model.fit(x=train_x, y=train_y.reshape(-1,1), validation_data=(val_x, val_y.reshape(-1,1)))
# #
# # # model prediction
p = model.predict(x=train_x, y=train_y)
val_p = model.predict(x=val_x, y=val_y.reshape(-1,1))
