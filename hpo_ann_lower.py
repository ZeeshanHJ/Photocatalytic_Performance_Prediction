
import site
site.addsitedir("D:\\AI4Water")

import json
import os
import math

import numpy as np
import pandas as pd
from SeqMetrics import RegressionMetrics
from ai4water import Model
from ai4water.models import MLP, LSTM


from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.hyperopt import HyperOpt, Categorical, Real, Integer
from sklearn.model_selection import train_test_split

from utils import PrepareData

# data preparation

inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

target = "Efficiency"
transformation = "ohe"     # le #ohe
lookback = 0
num_iterations = 100
return_sequences = False


prepare_data = PrepareData()

x,y = prepare_data(inputs,
                   target=target,
                   transformation=transformation,
                   lookback=lookback,
                   return_sequences=return_sequences,
                   )

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)



SEP = os.sep
PREFIX = f"{target}_{transformation}_{lookback}_hpo_{dateandtime_now()}"
ITER = 0

# these seeds are randomly generated but we keep track of the seed
# used at each iteration, so that when we rebuilt the model with optimized
# hyperparameters, we get reproducible results
SEEDS = np.random.randint(0, 1000, num_iterations)
# to keep track of seed being used at every optimization iteration
SEEDS_USED = []


def objective_fn(
        prefix=None,
        return_model=False,
        seed: int = None,
        **suggestions)->float:
    """This function must build, train and evaluate the ML model.
    The output of this function will be minimized by optimization algorithm.
    """
    suggestions = jsonize(suggestions)
    global ITER

    lr = suggestions.pop("lr")
    batch_size = suggestions.pop("batch_size")

    if lookback == 0:
        input_shape = train_x.shape[-1],
    else:
        input_shape = (lookback, train_x.shape[-1])

    model_config = MLP(
        input_shape=input_shape,
        output_features=1,
        **suggestions,
    )

    # build model
    _model = Model(
        model=model_config,
        input_features=inputs,
        output_features=[target],
          prefix=prefix or PREFIX,
          train_fraction=1.0,
          split_random=True,
          verbosity=0,
        epochs=1000,
        patience=800,
        batch_size=batch_size,
        ts_args={"lookback": lookback},
        x_transformation="minmax",
        lr=lr,
                    #y_transformation={"robust"},
                  )

    # ai4water's Model class does not fix numpy seed
    # below we fix all the seeds including numpy but this seed it itself randomly generated
    if seed is None:
        seed = SEEDS[ITER]
        SEEDS_USED.append(seed)

    _model.seed_everything(seed)

    # train model
    _model.fit(
        x=train_x, y=train_y.reshape(-1,1),
        validation_data=(val_x, val_y)
    )

    # evaluate model
    t, p = _model.predict(x=val_x, y=val_y.reshape(-1,1), return_true=True,
                          process_results=False)
    val_score = RegressionMetrics(t, p).r2()

    if not math.isfinite(val_score):
        val_score = -1.0

    # since the optimization algorithm solves minimization algorithm
    # we have to subtract r2_score from 1.0
    # if our validation metric is something like mse or rmse,
    # then we don't need to subtract it from 1.0
    val_score = 1.0 - val_score

    ITER += 1

    print(f"{ITER} {val_score}")

    if return_model:
        return _model

    return val_score

num_samples = 10

# parameter space
param_space = [
    Integer(16, 50, name="units"),
    Categorical(["relu", "elu", "tanh", "sigmoid"], name="activation"),
    Real(0.00001, 0.01, name="lr"),
    Categorical([8, 16, 32, 64], name="batch_size"),
    Integer(1, 5, name="num_layers")
]

x0 = [32, "relu", 0.001, 32, 2]

#
# optimizer = HyperOpt(
#     algorithm="bayes",
#     objective_fn=objective_fn,
#     param_space=param_space,
#     x0=x0,
#     num_iterations=num_iterations,
#     process_results=True,
#     opt_path=f"results{SEP}{PREFIX}",
#     verbosity=0,
# )
#
# results = optimizer.fit()
#
# print(optimizer.best_xy())

fpath = r'D:\Zeeshan\modeling\results\Efficiency_ohe_0_hpo_ann_20220518_135105\1_20220518_151507'
fpath = os.path.join(fpath,  "config.json")
model= Model.from_config_file(fpath)
model.verbosity = 1
w_path = os.path.join(model.path, "weights", "weights_923_1.89222.hdf5")
model.update_weights(w_path)

# train_p = model.predict(x=train_x, y=train_y, process_results=False,
#               plots=['residual', 'regression', 'prediction', 'murphy']
#               )
#
val_pred= model.predict(x=val_x,
              y=val_y,
              process_results=False,
            plots=['residual', 'regression', 'prediction', 'murphy']
              )

total_pred= model.predict(x=x,
              y=y,
              process_results=True,
            plots=['residual', 'regression', 'prediction', 'murphy']
              )

all_metrics = RegressionMetrics(y, total_pred).calculate_all()
all_metrics = jsonize(all_metrics)
with open(os.path.join(model.path, "total_errors.json"), "w") as fp:
    json.dump(all_metrics, fp)

fpath = os.path.join(model.path, "total_errors.csv")
pd.DataFrame.from_dict(all_metrics, orient='index').to_csv(fpath)

