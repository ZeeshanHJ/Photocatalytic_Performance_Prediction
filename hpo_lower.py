
import site
site.addsitedir("D:\\AI4Water")

import os
import math
from typing import Union

import numpy as np

from skopt.plots import plot_objective
from SeqMetrics import RegressionMetrics


from ai4water.datasets import busan_beach
from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.hyperopt import HyperOpt, Categorical, Real, Integer
from sklearn.model_selection import train_test_split

from utils import prepare_mg_dye_data, MyModel

# data preparation

inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

x,y = prepare_mg_dye_data(inputs, target="k")

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)


SEP = os.sep
PREFIX = f"hpo_{dateandtime_now()}"
ITER = 0

# sphinx_gallery_thumbnail_number = 2


def objective_fn(
        prefix=None,
        **suggestions)->float:
    """This function must build, train and evaluate the ML model.
    The output of this function will be minimized by optimization algorithm.
    """
    suggestions = jsonize(suggestions)
    global ITER

    # build model
    model = MyModel(model={"LassoLarsCV": suggestions},
                  prefix=prefix or PREFIX,
                  train_fraction=1.0,
                  split_random=True,
                  verbosity=0,
                    #y_transformation={"robust"},
                  )

    # train model
    model.fit(x=train_x, y=train_y)

    # evaluate model
    t, p = model.predict(x=val_x, y=val_y, return_true=True, process_results=False)
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

    return val_score

num_samples = 10

search_space = [
    # maximum number of trees that can be built
    Categorical(categories=[True, False], name='fit_intercept'),
Integer(low=100, high=1000, name='max_iter', num_samples=num_samples),
    Integer(low=500, high=5000, name='max_n_alphas', num_samples=num_samples)]
x0 = [True, 500, 1000]

# Now instantiate the HyperOpt class and call .fit on it
# algorithm can be either ``random``, ``grid``, ``bayes``, ``tpe``, ``bayes_rf``
#

optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_fn,
    param_space=search_space,
    x0=x0,
    num_iterations=100,
    process_results=True,
    opt_path=f"results{SEP}{PREFIX}",
    verbosity=0,
)

results = optimizer.fit()

print(optimizer.best_xy())