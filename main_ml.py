"""
This file shows application of a gradient boosting decision
tree method for efficiency prediction
"""

import site
site.addsitedir("D:\\mytools\\AI4Water")

import ai4water
print(ai4water.__version__)


import numpy as np
np.set_printoptions(suppress=True, linewidth=150)
from sklearn.model_selection import train_test_split

from utils import PrepareData, MyModel

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

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)


model = MyModel(
    model = {
            "CatBoostRegressor": {
                "iterations": 5000,
                "learning_rate": 0.49999999999999994,
                "l2_leaf_reg": 5.0,
                "model_size_reg": 0.1,
                "rsm": 0.743606499290401,
                "border_count": 1032,
                "feature_border_type": "UniformAndQuantiles",
                "logging_level": "Silent",
                "random_seed": 313
            }
        },
    input_features=prepare_data.inputs,
    output_features=['Efficiency'],
    split_random=True,
    train_fraction=1.0,
    val_fraction=0.3,
    x_transformation = "minmax",
)


# # model Training
h = model.fit(x=train_x, y=train_y)

print(model.evaluate(x=val_x, y=val_y,
                     metrics=["mse", 'r2']))

# # model prediction
p = model.predict(x=train_x, y=train_y)
p = model.predict(x=val_x, y=val_y)
