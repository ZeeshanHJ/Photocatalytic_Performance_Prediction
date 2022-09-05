
import site
site.addsitedir("D:\\mytools\\AI4Water")

from ai4water.utils.utils import jsonize
from ai4water.experiments import MLRegressionExperiments
from sklearn.model_selection import train_test_split
from ai4water.utils.utils import dateandtime_now

from utils import PrepareData, MyModel

# data preparation

inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

target = "Efficiency" #Efficiency #k
transformation = "ohe"# le #ohe
lookback = 0
run_tye="dry_run"  # optimize # dry_run
num_iterations=100

exp_name = f"{target}_{transformation}_{lookback}_{run_tye}_{dateandtime_now()}"

prepare_data = PrepareData()
x,y = prepare_data(inputs, target=target, transformation=transformation,
                          lookback=lookback)

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)


class MyExperiments(MLRegressionExperiments):

    def _build(self, title=None, **suggested_paras):
        """Builds the ai4water Model class"""

        suggested_paras = jsonize(suggested_paras)

        verbosity = max(self.verbosity-1, 0)
        if 'verbosity' in self.model_kws:
            verbosity = self.model_kws.pop('verbosity')

        model = MyModel(
            prefix=title,
            verbosity=verbosity,
            **self.model_kws,
            **suggested_paras
        )

        setattr(self, 'model_', model)
        return


# Define Experiment
comparisons = MyExperiments(
      input_features=prepare_data.inputs,
    output_features=[target],

      split_random=True,
    val_metric="r2",
    exp_name = exp_name,
    x_transformation="minmax",

)

# Training all models
comparisons.fit(x=train_x,
                y=train_y,
                validation_data=(val_x, val_y),
                run_type=run_tye,
                num_iterations=num_iterations,
                exclude=["LarsCV", "LassoLarsIC", "RANSACRegressor", "DummyRegressor", "OneClassSVM", "SGDRegressor", "TheilsenRegressor"]
                #include=["RandomForestRegressor"],
                )

# postprocessing of results
comparisons.compare_errors('r2')

# find out the models which resulted in r2> 0.5
best_models = comparisons.compare_errors('r2', cutoff_type='greater',
                                               cutoff_val=0.5)
comparisons.compare_errors('corr_coeff')

best_models = comparisons.compare_errors('mse', cutoff_type='lower',
                                               cutoff_val=0.5)

best_models = comparisons.compare_errors('rmse', cutoff_type='lower',
                                               cutoff_val=0.5)

best_models = comparisons.compare_errors('mae', cutoff_type='lower',
                                               cutoff_val=0.5)

comparisons.compare_errors('mape', cutoff_type="lower", cutoff_val=300)

comparisons.compare_errors('nrmse', cutoff_type="lower", cutoff_val=2)

# comparisons.compare_errors('r2')
comparisons.taylor_plot()  # see help(comparisons.taylor_plot()) to tweak the taylor plot