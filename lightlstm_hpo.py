
import site
site.addsitedir("D:\\AI4Water")

import os
import math
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SeqMetrics import RegressionMetrics
from easy_mpl import regplot
from tensorflow.keras.layers import Input, Dense, Flatten


from ai4water.functional import Model
from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.hyperopt import HyperOpt, Categorical, Real, Integer
from sklearn.model_selection import train_test_split

from lightlstm import LightLSTM
from utils import PrepareData

# data preparation
inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

target = "Efficiency"
transformation = "ohe"     # le #ohe
lookback = 10
num_iterations = 100
return_sequences = True

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


SEP = os.sep
PREFIX = f"{target}_{transformation}_{lookback}_hpo_{dateandtime_now()}"
ITER = 0

# these seeds are randomly generated but we keep track of the seed
# used at each iteration, so that when we rebuilt the model with optimized
# hyperparameters, we get reproducible results
SEEDS = np.random.randint(0, 1000, num_iterations)
# to keep track of seed being used at every optimization iteration
SEEDS_USED = []

batch_size = 4
num_dyn_inputs = 1
num_static_inputs = 28

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

    units = suggestions['units']
    activation = suggestions['activation']

    _model = FModel(
        model = {"layers": {"units": units, "activation": activation}},
        lr= suggestions['lr'],
        patience=800,
        epochs=1000,
        verbosity=0,
        prefix=prefix or PREFIX,
    )

    if seed is None:
        seed = SEEDS[ITER]
        SEEDS_USED.append(seed)

    _model.seed_everything(seed)

    _model.fit(
        x=[train_x_dyn[0:92], train_x_static[0:92]], y=train_y[0:92].reshape(92, -1),
        validation_data=([val_x_dyn[0:40], val_x_static[0:40]], val_y[0:40].reshape(40, -1)),
        batch_size=batch_size,

    )

    p = _model.predict([val_x_dyn[0:40], val_x_static[0:40]])

    val_score = RegressionMetrics(val_y[0:40].reshape(-1,1), p.reshape(-1,1)).r2()

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
    Integer(16, 100, name="units"),
    Categorical(["relu", "elu", "tanh", "sigmoid"], name="activation"),
    Real(0.00001, 0.01, name="lr"),
    #Categorical([8, 16, 32, 64], name="batch_size"),
    #Integer(1, 5, name="num_layers")
]

x0 = [32, "relu", 0.001]



optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_fn,
    param_space=param_space,
    x0=x0,
    num_iterations=num_iterations,
    process_results=True,
    opt_path=f"results{SEP}{PREFIX}",
    verbosity=0,
)

results = optimizer.fit()

print(optimizer.best_xy())


fpath = os.path.join(os.getcwd(), optimizer.opt_path, "1_20220519_162029",  "config.json")
model = FModel.from_config_file(fpath)
model.verbosity = 1
w_path = os.path.join(model.path, "weights", "weights_999_10.48209.hdf5")
model.update_weights(w_path)

train_p = model.predict(
    x=[train_x_dyn[0:92], train_x_static[0:92]],
     y=train_y[0:92].reshape(92, -1),
     process_results=False
     )

# regression plot
ax = regplot(train_y[0:92].reshape(-1,1), train_p.reshape(-1,1), show=False)
plt.savefig(os.path.join(model.path, "train_reg_plot.png"), dpi=300)

# save errors
train_metrics = RegressionMetrics(
    train_y[0:92].reshape(-1,1), train_p.reshape(-1,1)).calculate_all()
train_metrics = jsonize(train_metrics)
with open(os.path.join(model.path, "train_errors.json"), "w") as fp:
    json.dump(train_metrics, fp)

fpath = os.path.join(model.path, "train_errors.csv")
pd.DataFrame.from_dict(train_metrics, orient='index').to_csv(fpath)


test_p = model.predict(
    x=[val_x_dyn[0:40], val_x_static[0:40]],
    y=val_y[0:40].reshape(40, -1),
    process_results=False
)

# regression plot
ax = regplot(val_y[0:40].reshape(-1,1), test_p.reshape(-1,1), show=False)
plt.savefig(os.path.join(model.path, "test_reg_plot.png"), dpi=300)

# save errors
test_metrics = RegressionMetrics(
    val_y[0:40].reshape(-1,1), test_p.reshape(-1,1)).calculate_all()
test_metrics = jsonize(test_metrics)
with open(os.path.join(model.path, "test_errors.json"), "w") as fp:
    json.dump(test_metrics, fp)

fpath = os.path.join(model.path, "test_errors.csv")
pd.DataFrame.from_dict(test_metrics, orient='index').to_csv(fpath)


# TOTOAL DATA
new_samples = 1
dummy_x = np.zeros((new_samples, 10, 29))
dummy_y = np.zeros((new_samples, 10))
new_x = np.vstack([x, dummy_x])
new_y = np.vstack([y, dummy_y])

xx = [np.expand_dims(new_x[..., -1], axis=-1), new_x[..., 0:-1]]
total_pred = model.predict(
    x=xx,
    y=new_y,
    process_results=False,
)

total_pred1 = total_pred[0:-1].reshape(-1, 1)
total_true1 = new_y[0:-1].reshape(-1, 1)
print(RegressionMetrics(total_true1, total_pred1).r2())


total_data = np.hstack([x.reshape(-1, 29),
                        total_pred1,
                        total_true1
                        ])

total_data = pd.DataFrame(total_data,
                          columns=prepare_data.inputs + [f"pred_{target}", f"true_{target}"])

total_data.to_csv(os.path.join(model.path, "total_data.csv"))

ax = regplot(total_true1, total_pred1, show=False)
plt.savefig(os.path.join(model.path, "reg_plot.png"), dpi=300)

# save errors
all_metrics = RegressionMetrics(total_true1, total_pred1).calculate_all()
all_metrics = jsonize(all_metrics)
with open(os.path.join(model.path, "total_errors.json"), "w") as fp:
    json.dump(all_metrics, fp)

fpath = os.path.join(model.path, "total_errors.csv")
pd.DataFrame.from_dict(all_metrics, orient='index').to_csv(fpath)