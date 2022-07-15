import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from shap import Explanation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ai4water.postprocessing import ShapExplainer

from utils import PrepareData, MyModel

c_path = r"D:\Old system data\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\CatBoostRegressor\best\20220512_115620"
model = MyModel.from_config_file(os.path.join(c_path, "config.json"))
model.config['verbosity'] = 1
model.verbosity = 1
model.update_weights(os.path.join(c_path, "weights", "CatBoostRegressor"))

inputs = ['Catalyst_type', 'Catalyst_loading',
          'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci',
          'Surface area', 'Pore Volume'
          ]

target = "Efficiency" #Efficiency #k
transformation = "ohe"# le #ohe
lookback = 0
run_tye="dry_run"  # optimize # dry_run
num_iterations=100

prepare_data = PrepareData()
x,y = prepare_data(inputs,
                   target=target,
                   transformation=transformation,
                   lookback=lookback,
                            )

new_inputs = prepare_data.inputs
cat_encoder = prepare_data.cat_encoder
anion_encoder = prepare_data.anion_encoder

train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)

# TreeExplainer
from shap.plots import heatmap, beeswarm, scatter, bar
explainer = shap.Explainer(model._model, feature_names=new_inputs)
shap_values = explainer(val_x)

catalyst_sv = shap_values.values[:, 8:23].sum(axis=1)
anion_sv = shap_values.values[:, 23:].sum(axis=1)
shap_values_new = np.column_stack([shap_values.values[:, 0:8], catalyst_sv, anion_sv])

shap_values.values = shap_values_new
shap_values.feature_names = new_inputs[0:8] + ["Catalyst", "Anions"]
cat_enc = val_x[:, 8:23]
cat_dec = prepare_data.cat_encoder.inverse_transform(cat_enc)
an_enc = val_x[:, 23:]
an_dec = prepare_data.anion_encoder.inverse_transform(an_enc)
# because categorical values can not be plotted properly, this time we
# label encode them, this is only for plotting purpose
cat_encoder1 = LabelEncoder()
cat_enc1 = cat_encoder1.fit_transform(cat_dec.reshape(-1, ))
an_encoder1 = LabelEncoder()
an_enc1 = an_encoder1.fit_transform(an_dec.reshape(-1, ))
new_data = np.column_stack([val_x[:, 0:8], cat_enc1, an_enc1])
shap_values.data = new_data
cat_labels = {k:v for k,v in zip(range(len(cat_encoder1.classes_)), cat_encoder1.classes_)}
with open(os.path.join(c_path, "shap", "cat_labels.json"), "w") as fp:
    json.dump(cat_labels, fp)
an_labels = {k:v for k,v in zip(range(len(an_encoder1.classes_)), an_encoder1.classes_)}
with open(os.path.join(c_path, "shap", "an_labels.json"), "w") as fp:
    json.dump(an_labels, fp)

#
plt.close('all')
heatmap(shap_values)
plt.savefig(os.path.join(c_path, "shap", "heatmap.png"), bbox_inches="tight", dpi=300)

plt.close('all')
heatmap(shap_values, instance_order=shap_values.sum(1))
plt.savefig(os.path.join(c_path, "shap", "heatmap_sum.png"), bbox_inches="tight", dpi=300)

plt.close('all')
heatmap(shap_values, feature_values=shap_values.abs.max(0))
plt.savefig(os.path.join(c_path, "shap", "heatmap_abs_max.png"), bbox_inches="tight", dpi=300)

plt.close('all')
beeswarm(shap_values)
plt.savefig(os.path.join(c_path, "shap", "beeswarm.png"), bbox_inches="tight", dpi=300)

plt.close('all')
scatter(shap_values)
plt.savefig(os.path.join(c_path, "shap", "scatter.png"), bbox_inches="tight", dpi=300)

plt.close('all')
bar(shap_values)
plt.savefig(os.path.join(c_path, "shap", "bar.png"), bbox_inches="tight", dpi=300)

plt.close('all')
shap.summary_plot(shap_values, plot_type="violen")
plt.savefig(os.path.join(c_path, "shap", "summary_violen.png"), bbox_inches="tight", dpi=300)

plt.close('all')
shap.summary_plot(shap_values, plot_type="dot")
plt.savefig(os.path.join(c_path, "shap", "summary_dot.png"), bbox_inches="tight", dpi=300)

plt.close('all')
shap.force_plot(explainer.expected_value, shap_values.values[0].reshape(-1,),
                new_data[0].reshape(-1,),
                feature_names=shap_values.feature_names,
                matplotlib=True)
plt.show()

for idx in [57, 66, 95, 45]:
    plt.close('all')
    e = Explanation(
        shap_values.values[idx].reshape(-1,),
        base_values=explainer.expected_value,
        data=new_data[idx].reshape(-1,),
        feature_names=shap_values.feature_names,
    )
    shap.plots.waterfall(e, show=False)
    plt.savefig(os.path.join(c_path, "shap", f"waterfall_{idx}.png"), bbox_inches="tight", dpi=300)



from easy_mpl import hist, plot
ax = hist(new_data[:, 1], show=False)
ax2 = ax.twiny()
plot(shap_values.values[:, 1], 'o', show=False, ax=ax2)
plt.show()

# Tree Expaliner
# tree_explainer = shap.TreeExplainer(model._model)
# tree_explainer.model.original_model.params = model._model.get_params()
# new_data_df = pd.DataFrame(new_data, columns=new_inputs[0:8] + ["Catalyst", "Anions"])
# for col in new_inputs[0:8]:
#     new_data_df[col] = new_data_df[col].astype("float64")
# for col in ["Catalyst", "Anions"]:
#     new_data_df[col] = new_data_df[col].astype("category")
# shap_values_result = tree_explainer.shap_values(new_data_df, y=val_y)