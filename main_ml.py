import site
site.addsitedir("D:\\AI4Water")

import ai4water
print(ai4water.__version__)


import numpy as np
np.set_printoptions(suppress=True, linewidth=150)
import matplotlib.pyplot as plt
import pandas as pd
from easy_mpl import plot, imshow
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ai4water.hyperopt import Categorical, Real, Integer



from utils import prepare_mg_dye_data, MyModel

inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

x,y = prepare_mg_dye_data(
    inputs, target="k",
    transformation="ohe",
)

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)


model = MyModel(
    model= "LGBMRegressor", #{"CatBoostRegressor": {
        #'iterations': 1000,
        #'learning_rate':0.01,
        #'l2_leaf_reg':3.0,
        #'model_size_reg':0.5,
        #'rsm': 0.5,
        #'border_count': 32,
        #'feature_border_type': 'GreedyLogSum',
        #'n_estimators': 800,
        #'max_depth': 1,
        #'learning_rate': 0.4,
       #'booster': "gbtree"
    #}},
    y_transformation="minmax",
    output_features=["k"]
)


# # model Training
h = model.fit(x=train_x, y=train_y)

print(model.evaluate(x=val_x, y=val_y,
                     metrics=["mse", 'r2']))
# #
# # # model prediction
# p = model.predict(x=train_x, y=train_y)
# p = model.predict(x=val_x, y=val_y)
#
# plt.close('all')
#
# imshow(model._model.feature_importances_.reshape(10, 10))