
import site
from typing import Union

site.addsitedir("D:\\AI4Water")
site.addsitedir("D:\\AutoTab")

from sklearn.model_selection import train_test_split

from autotab import OptimizePipeline

from utils import MyModel, prepare_mg_dye_data


inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

x,y = prepare_mg_dye_data(inputs, target="k")

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)

class MyPipeline(OptimizePipeline):


    def _build_model(
            self,
            model,
            val_metric: str,
            x_transformation,
            y_transformation,
            prefix: Union[str, None],
            verbosity:int = 0,
            batch_size:int = 32,
            lr:float = 0.001,
    ) -> MyModel:
        """
        build the ai4water Model. When overwriting this method, the user
        must return an instance of ai4water's Model_ class.

        Parameters
        ----------
            model :
                anything which can be fed to AI4Water's Model class.
            x_transformation :
            y_transformation :
            prefix :
            verbosity :
            batch_size :
                only used when category is "DL".
            lr :
                only used when category is "DL"

        .. Model:
            https://ai4water.readthedocs.io/en/master/model.html#ai4water._main.BaseModel
        """
        model = MyModel(
            model=model,
            verbosity=verbosity,
            val_metric=val_metric,
            x_transformation=x_transformation,
            y_transformation=y_transformation,
            # seed=self.seed,
            prefix=prefix,
            batch_size=int(batch_size),
            lr=float(lr),
            **self.model_kwargs
        )
        return model


pl = MyPipeline(
    inputs_to_transform=[],
    outputs_to_transform=[],

    parent_iterations=14,
    child_iterations=50,

    child_algorithm="random",

    models=["XGBRegressor", "RandomForestRegressor"],

    monitor=["r2", "mae", "nrmse"]

)

res = pl.fit(x=x,y=y, validation_data=(val_x, val_y))