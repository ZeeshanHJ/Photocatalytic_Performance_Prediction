import os

import site
site.addsitedir("D:\\AI4Water")

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from easy_mpl import bar_chart
from SeqMetrics import RegressionMetrics
from pdp_box.info_plots import actual_plot, actual_plot_interact

from ai4water.utils.utils import dateandtime_now, jsonize

from utils import PrepareData, MyModel, prediction_dist_heatmap, decode_with_ohe
from heatmap import heatmap, annotate_heatmap

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
x,y = prepare_data(inputs,
                          target=target,
                          transformation=transformation,
                          lookback=lookback,
                            )

new_inputs = prepare_data.inputs
cat_encoder = prepare_data.cat_encoder
anion_encoder = prepare_data.anion_encoder


train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.3, random_state=313)

c_path = r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\CatBoostRegressor\best\20220512_115620"
model = MyModel.from_config_file(os.path.join(c_path, "config.json"))

model.verbosity = 1
model.update_weights()


train_true, train_pred = model.predict(
    x=train_x,
    y=train_y,
    return_true=True,
    process_results=False,
    plots=["residual", "regression", "prediction", "murphy"],
                    )

val_true, val_pred = model.predict(
    x=val_x,
    y=val_y,
    return_true=True,
    process_results=False,
    plots=["residual", "regression", "prediction", "murphy"],
                    )

# making predictions on all unsorted data and saving the results
total_true, total_pred = model.predict(
    x=x, y=y,
    return_true=True,
    process_results=False,
    plots=["residual", "regression", "prediction", "murphy"],
)

anion_cols = [col for col in prepare_data.inputs if 'Anions_' in col]
an_data_enc = x[:, -len(anion_cols):]
an_data_dec = anion_encoder.inverse_transform(an_data_enc)

catalyst_cols = [col for col in prepare_data.inputs if 'catalyst_' in col]
cat_data_enc = x[:, 8:-6]
cat_data_dec = cat_encoder.inverse_transform(cat_data_enc)

new_x = np.concatenate([x[:, 0:8], cat_data_dec, an_data_dec], axis=1)

total_data = np.hstack([new_x, total_true, total_pred.reshape(-1,1)])
total_io = pd.DataFrame(
    total_data,
    columns=prepare_data.inputs[0:8] + ["Catalyst_type", "Anions", f"true_{target}", f"pred_{target}"])
#total_io.to_csv(os.path.join(model.path, "total_data1.csv"))
#
# all_metrics = RegressionMetrics(total_true, total_pred).calculate_all()
# all_metrics = jsonize(all_metrics)
# with open(os.path.join(model.path, "total_errors.json"), "w") as fp:
#     json.dump(all_metrics, fp)
#
# fpath = os.path.join(model.path, "total_errors.csv")
# pd.DataFrame.from_dict(all_metrics, orient='index').to_csv(fpath)
##################################################################################################

train_x_df = pd.DataFrame(train_x, columns=new_inputs)
train_io = train_x_df.copy()
train_io['true'] = train_true
train_io['predicted'] = train_pred
#train_io.to_csv(os.path.join(model.path, "train_io.csv"))


val_x_df = pd.DataFrame(val_x, columns=new_inputs)
val_io = val_x_df.copy()
val_io['true'] = val_true
val_io['predicted'] = val_pred


total_x_df = pd.concat([train_x_df, val_x_df])

# change column names to original one
mapper = {k:v for k,v in zip(catalyst_cols, cat_encoder.categories_[0])}
total_x_df = total_x_df.rename(columns=mapper)

mapper = {k:v for k,v in zip(anion_cols, anion_encoder.categories_[0])}
total_x_df = total_x_df.rename(columns=mapper)

# fig, axes, summary_df = actual_plot(
# model=model, X=total_x_df, feature='Catalyst_loading', feature_name='Catalyst_loading',
# cust_grid_points=[0, 1.,  1.6, 2.5]
# )
# plt.savefig(os.path.join(model.path, "catalyst_loading.png"), dpi=500)
# plt.show()
#
# # for categroical features
# fig, axes, summary_df = actual_plot(
# model=model, X=total_x_df,
# feature=[f'catalyst_{i}' for i in range(15)],
# feature_name='Catalyst_type',
# )
# plt.savefig(os.path.join(model.path, "catalyst_type.png"), dpi=500)
# plt.show()
#
# fig, axes, summary_df = actual_plot(
# model=model, X=total_x_df,
# feature=[f'Anions_{i}' for i in range(6)],
# feature_name='Anions',
# )
# plt.savefig(os.path.join(model.path, "Anions.png"), dpi=500)
# plt.show()
#
# fig, axes, summary_df = actual_plot(
# model=model, X=total_x_df, feature='Light_intensity', feature_name='Light_intensity',
# cust_grid_points=[0, 1, 26, 56, 105]
# )
# plt.savefig(os.path.join(model.path, "Light_intensity.png"), dpi=500)
# plt.show()
#
# fig, axes, summary_df = actual_plot(
# model=model, X=total_x_df, feature='Ci', feature_name='Ci',
# cust_grid_points=[5, 10, 20, 40, 80]
# )
# plt.savefig(os.path.join(model.path, "Ci.png"), dpi=500)
# plt.show()
#
# fig, axes, summary_df = actual_plot(
# model=model, X=total_x_df, feature='solution_pH', feature_name='solution_pH',
# cust_grid_points=[3, 4.9, 5.1, 5.5, 7.1, 9]
# )
# plt.savefig(os.path.join(model.path, "solution_pH.png"), dpi=500)
# plt.show()
#
# fig, axes, summary_df = actual_plot(
# model=model, X=total_x_df, feature='time', feature_name='time',
# cust_grid_points=[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271]
# )
# plt.savefig(os.path.join(model.path, "time.png"), dpi=500)
# plt.show()
#
# fig, axes, summary_df = actual_plot(
# model=model, X=total_x_df, feature='Surface area', feature_name='Surface area',
# cust_grid_points=[0, 10, 20, 30, 50]
# )
# plt.savefig(os.path.join(model.path, "Surface area.png"), dpi=500)
# plt.show()



#fig, axes, summary_df = actual_plot(
#model=model, X=total_x_df, feature='Pore Volume', feature_name='Pore Volume',
#cust_grid_points=[0, 0.0015, 0.0025, 0.0035, 0.0045, 0.0051]
#)
#plt.savefig(os.path.join(model.path, "Pore Volume.png"), dpi=500)
#plt.show()

#fig, axes, summary_df = actual_plot(
#model=model, X=total_x_df, feature='HA', feature_name='HA',
#cust_grid_points=[0, 2, 5, 7, 10]
#)
#plt.savefig(os.path.join(model.path, "HA.png"), dpi=500)
#plt.show()

#
#fig, axes, summary_df = actual_plot_interact(
#model=model, X=total_x_df,
#features=['Catalyst_loading', 'Light_intensity'],
#feature_names=['Catalyst_loading', 'Light_intensity'],
#cust_grid_points=[[0, 1., 1.6, 2.5],
#[0, 1, 26, 56, 105]]
#)
#plt.savefig(os.path.join(model.path, "catalyst_loading_light.png"), dpi=500)
#plt.show()

#prediction_dist_heatmap(summary_df,
#                       font_cmap=None,
#                      annot_fontsize=6,
#                     cmap="tab20",
#                    xlabel=None,
#                   ylabel=None,
#                  )
#plt.savefig(os.path.join(model.path, "catalyst_loading_light_heatmap.png"), dpi=500)
#plt.show()

#***************** INTERACTION******************
# # for categroical features
# fig, axes, summary_df = actual_plot_interact(
#   model=model, X=total_x_df,
#  features=['time', cat_encoder.categories_[0].tolist()],
# feature_names=['time', 'Catalyst_type'],
#     annotate=True,
#     annotate_counts=False,
#     cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271], None]
# )
# plt.savefig(os.path.join(model.path, "time_catalyst_type.png"), dpi=500)
# plt.show()
#
# yticklabels = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
# df, _ = prediction_dist_heatmap(
#     summary_df,
#     font_cmap=None,
#     annot_fontsize=6,
#     cmap="tab20",
#     xlabel=None,
#     ylabel=None,
#     yticklabels=yticklabels
#                  )
# plt.savefig(os.path.join(model.path, "time_catalyst_type_heatmap.png"), dpi=500)
# plt.show()
# df2 = pd.DataFrame()
# for col in ['no catalyst', 'pure BFO', 'commercial TiO2', '1 wt% Ag-BFO',
#             '2 wt% Ag-BFO', '3 wt% Ag-BFO', '4 wt% Ag-BFO', '0.25 wt% Pt-BFO',
#             '0.5 wt% Pt-BFO', '1 wt% Pt-BFO', '2 wt% Pt-BFO', '0.5 wt% Pd-BFO',
#             '1 wt% Pd-BFO', '2 wt% Pd-BFO', '3 wt% Pd-BFO']:
#     df2[col] = df[col]
#
# fig, ax = plt.subplots()
#
# im, cbar = heatmap(df2.values, df.index, df2.columns, ax=ax,
#                    cmap="YlGn", cbarlabel="Efficiency", xlabel_on_top=True)
# texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=7)
#
# fig.tight_layout()
# plt.savefig(os.path.join(model.path, "time_catalyst_type_heatmap1.png"), dpi=500)
# plt.show()


# # time vs light intensity
fig, axes, summary_df = actual_plot_interact(
  model=model, X=total_x_df,
 features=['time', "Light_intensity"],
feature_names=['time', 'Light intensity'],
    annotate=True,
    annotate_counts=False,
    cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271],
                      [0, 24, 54, 104, 106]]
)
plt.savefig(os.path.join(model.path, "time_light_intensity.png"), dpi=500)
plt.show()

yticklabels = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
df, _ = prediction_dist_heatmap(
    summary_df,
    font_cmap=None,
    annot_fontsize=6,
    cmap="tab20",
    xlabel=None,
    ylabel=None,
    yticklabels=yticklabels
                 )
plt.savefig(os.path.join(model.path, "time_light_intensity_heatmap.png"), dpi=500)
plt.show()
df2 = pd.DataFrame()
for col in df.columns:
    df2[col] = df[col]

fig, ax = plt.subplots()

im, cbar = heatmap(df2.values, df.index, df2.columns, ax=ax,
                   cmap="YlGn", cbarlabel="Efficiency", xlabel_on_top=True)
texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=7)

fig.tight_layout()
plt.savefig(os.path.join(model.path, "time_light_intensity_heatmap1.png"), dpi=500)
plt.show()


# # # for categroical features
# fig, axes, summary_df = actual_plot_interact(
# model=model, X=total_x_df,
# features=['time', anion_encoder.categories_[0].tolist()],
# feature_names=['time', 'Anions'],
# annotate=True,
# annotate_counts=False,
# cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271], None]
# )
# plt.savefig(os.path.join(model.path, "time_anions.png"), dpi=500)
#
# plt.show()
#
# df, _ = prediction_dist_heatmap(summary_df,
#                       font_cmap=None,
#                      annot_fontsize=6,
#                     cmap="tab20",
#                    xlabel=None,
#                   ylabel=None,
#                                 yticklabels=yticklabels
#                  )
# plt.savefig(os.path.join(model.path, "time_Anions_heatmap.png"), dpi=500)
# plt.show()
# df2 = pd.DataFrame()
# for col in ['without Anion', 'NaHCO3', 'NaCO3', 'Na2SO4', 'Na2HPO4', 'NaCl']:
#     df2[col] = df[col]
#
# fig, ax = plt.subplots()
#
# im, cbar = heatmap(df2.values, df.index, df2.columns, ax=ax,
#                    cmap="YlGn", cbarlabel="Efficiency", xlabel_on_top=True)
# texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=7)
#
# fig.tight_layout()
# plt.savefig(os.path.join(model.path, "time_Anions_heatmap1.png"), dpi=500)
# plt.show()
#
#
# # for categroical features
# fig, axes, summary_df = actual_plot_interact(
# model=model, X=total_x_df,
# features=['time', 'solution_pH'],
# feature_names=['time', 'solution_pH'],
# annotate=True,
# annotate_counts=True,
# cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271],
#                   [0, 3.1, 5.1, 5.46, 7.1, 9.1]])
# plt.savefig(os.path.join(model.path, "time_solution_pH.png"), dpi=500)
# plt.show()
#
# yticklabels = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
# xticklabels = [0, 3, 5, 5.45, 7, 9]
#
# df, _ = prediction_dist_heatmap(summary_df,
#                       font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                                 yticklabels=yticklabels
#                       )
# plt.savefig(os.path.join(model.path, "time_solution_pH_heatmap.png"), dpi=500)
# plt.show()
#
# df2 = pd.DataFrame()
# for col in df.columns:
#     df2[col] = df[col]
#
# mapper = {k:v for k,v in zip(df2.columns[::-1], xticklabels[1:])}
# df2 = df2.rename(columns=mapper)
# fig, ax = plt.subplots()
#
# im, cbar = heatmap(df2.values, df.index, df2.columns, ax=ax,
#                    cmap="YlGn", cbarlabel="Efficiency", xlabel_on_top=True)
# texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=7)
#
# fig.tight_layout()
# plt.savefig(os.path.join(model.path, "time_solution_pH_heatmap1.png"), dpi=500)
# plt.show()
#
#
# # for categroical features
# fig, axes, summary_df = actual_plot_interact(
# model=model, X=total_x_df,
# features=['time', 'Ci'],
# feature_names=['time', 'Ci'],
# annotate=True,
# annotate_counts=False,
# cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271],
#                   [0,  5.1, 10.1, 20.1, 40.1, 80.1]]
# )
# plt.savefig(os.path.join(model.path, "time_Ci.png"), dpi=500)
# plt.show()
#
# xticklabels = [5, 10, 20, 40, 80]
#
# df, _ = prediction_dist_heatmap(summary_df,
#                        font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                                 yticklabels=yticklabels
#                       )
# plt.savefig(os.path.join(model.path, "time_Ci_heatmap.png"), dpi=500)
# plt.show()
#
# df2 = pd.DataFrame()
# for col in df.columns:
#     df2[col] = df[col]
#
# mapper = {k:v for k,v in zip(df2.columns[::-1], xticklabels)}
# df2 = df2.rename(columns=mapper)
# fig, ax = plt.subplots()
#
# im, cbar = heatmap(df2.values, df.index, df2.columns, ax=ax,
#                    cmap="YlGn", cbarlabel="Efficiency", xlabel_on_top=True)
# texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=7)
#
# fig.tight_layout()
# plt.savefig(os.path.join(model.path, "time_Ci_heatmap1.png"), dpi=500)
# plt.show()
#
#
#
# ## for categroical features
# fig, axes, summary_df = actual_plot_interact(
# model=model, X=total_x_df,
# features=['time', 'Catalyst_loading'],
# feature_names=['time', 'Catalyst_loading'],
# annotate=True,
# annotate_counts=False,
# cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271],
#                   [0, 0.1, 0.6, 1.1, 1.6, 2.1, 2.6]]
# )
# plt.savefig(os.path.join(model.path, "time_Catalyst_loading.png"), dpi=500)
# plt.show()
#
# yticklabels = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
# xticklabels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
# df, _ = prediction_dist_heatmap(summary_df,
#                        font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                         yticklabels=yticklabels
#                        )
# plt.savefig(os.path.join(model.path, "time_Catalyst_loading_heatmap.png"), dpi=500)
# plt.show()
# df2 = pd.DataFrame()
# for col in df.columns:
#     df2[col] = df[col]
#
# mapper = {k:v for k,v in zip(df2.columns[::-1], xticklabels)}
# df2 = df2.rename(columns=mapper)
# fig, ax = plt.subplots()
#
# im, cbar = heatmap(df2.values, df.index, df2.columns, ax=ax,
#                    cmap="YlGn", cbarlabel="Efficiency", xlabel_on_top=True)
# texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=7)
#
# fig.tight_layout()
# plt.savefig(os.path.join(model.path, "time_Catalyst_loading_heatmap1.png"), dpi=500)
# plt.show()
#
#
# ## for categroical features
# fig, axes, summary_df = actual_plot_interact(
# model=model, X=total_x_df,
# features=['time', 'HA'],
# feature_names=['time', 'HA'],
# annotate=True,
# annotate_counts=False,
# cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271],
#                   [0, 0.1, 2.1, 5.1, 7.1, 10.1]]
# )
# plt.savefig(os.path.join(model.path, "time_HA.png"), dpi=500)
# plt.show()
#
# xticklabels = [0, 2, 5, 7, 10]
# yticklabels = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
#
# df, _ = prediction_dist_heatmap(summary_df,
#                        font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                                 yticklabels=yticklabels
#                        )
# plt.savefig(os.path.join(model.path, "time_HA_heatmap.png"), dpi=500)
# plt.show()
#
# df2 = pd.DataFrame()
# for col in df.columns:
#     df2[col] = df[col]
#
# mapper = {k:v for k,v in zip(df2.columns[::-1], xticklabels)}
# df2 = df2.rename(columns=mapper)
# fig, ax = plt.subplots()
#
# im, cbar = heatmap(df2.values, df.index, df2.columns, ax=ax,
#                    cmap="YlGn", cbarlabel="Efficiency", xlabel_on_top=True)
# texts = annotate_heatmap(im, valfmt="{x:.1f}", fontsize=7)
#
# fig.tight_layout()
# plt.savefig(os.path.join(model.path, "time_HA_heatmap1.png"), dpi=500)
# plt.show()


# for categroical features
fig, axes, summary_df = actual_plot_interact(
model=model, X=total_x_df,
features=['time', 'Surface area'],
feature_names=['time', 'Surface area'],
annotate=True,
annotate_counts=False,
cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271],
                  None]
)
plt.savefig(os.path.join(model.path, "time_Surface_area.png"), dpi=500)
plt.show()

yticklabels = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
df, _ = prediction_dist_heatmap(summary_df,
                       font_cmap=None,
                       annot_fontsize=6,
                       cmap="tab20",
                       xlabel=None,
                       ylabel=None,
                                yticklabels=yticklabels
                       )
plt.savefig(os.path.join(model.path, "time_Surface_area1.png"), dpi=500)
plt.show()

#
# for categroical features
#fig, axes, summary_df = actual_plot_interact(
#model=model, X=total_x_df,
#features=['time', 'Pore Volume'],
#feature_names=['time', 'Pore Volume'],
#annotate=True,
#annotate_counts=False,
#cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271], None]
#)
#plt.savefig(os.path.join(model.path, "time_Pore Volume.png"), dpi=500)
#plt.show()
#prediction_dist_heatmap(summary_df,
#                        font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                        )
#plt.savefig(os.path.join(model.path, "time_Pore Volume_heatmap.png"), dpi=500)
#plt.show()

#
# for categroical features
#fig, axes, summary_df = actual_plot_interact(
#model=model, X=total_x_df,
#features=['Ci', 'Catalyst_loading'],
#feature_names=['Ci', 'Catalyst_loading'],
#annotate=True,
#annotate_counts=False,
#cust_grid_points=[[5, 10, 20, 40, 80], [0, 1.,  1.6, 2.5]])
#plt.savefig(os.path.join(model.path, "Ci_Catalyst_loading.png"), dpi=500)
#plt.show()

#prediction_dist_heatmap(summary_df,
#                        font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                        )
#plt.savefig(os.path.join(model.path, "Ci_Catalyst_loading_heatmap.png"), dpi=500)
#plt.show()

#
# for categroical features
#fig, axes, summary_df = actual_plot_interact(
#model=model, X=total_x_df,
#features=['Ci', 'solution_pH'],
#feature_names=['Ci', 'solution_pH'],
#annotate=True,
#annotate_counts=False,
#cust_grid_points=[[5, 10, 20, 40, 80], None]
#)
#plt.savefig(os.path.join(model.path, "Ci_solution_pH.png"), dpi=500)
#plt.show()

#prediction_dist_heatmap(summary_df,
#                        font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                        )
#plt.savefig(os.path.join(model.path, "Ci_solution_pH_heatmap.png"), dpi=500)
#plt.show()

#
# for categroical features
#fig, axes, summary_df = actual_plot_interact(
#model=model, X=total_x_df,
#features=['Light_intensity', 'Ci'],
#feature_names=['Light_intensity', 'Ci'],
#annotate=True,
#annotate_counts=False,
#cust_grid_points=[[0, 1, 26, 56, 105], [5, 10, 20, 40, 80]]
#)
#plt.savefig(os.path.join(model.path, "Light_intensity_Ci.png"), dpi=500)
#plt.show()

#prediction_dist_heatmap(summary_df,
#                        font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                        )
#plt.savefig(os.path.join(model.path, "Light_intensity_Ci_heatmap.png"), dpi=500)
#plt.show()
#
# for categroical features
#fig, axes, summary_df = actual_plot_interact(
#model=model, X=total_x_df,
#features=['time', 'Light_intensity'],
#feature_names=['time', 'Light_intensity'],
#annotate=True,
#annotate_counts=False,
#cust_grid_points=[[0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 271], [0, 1, 26, 56, 105]]
#)
#plt.savefig(os.path.join(model.path, "time_Light_intensity.png"), dpi=500)
#plt.show()
#prediction_dist_heatmap(summary_df,
#                        font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                        )
#plt.savefig(os.path.join(model.path, "time_Light_intensity_heatmap.png"), dpi=500)
#plt.show()

#
# for categroical features
#fig, axes, summary_df = actual_plot_interact(
#model=model, X=total_x_df,
#features=['Light_intensity', 'Catalyst_loading'],
#feature_names=['Light_intensity', 'Catalyst_loading'],
#annotate=True,
#annotate_counts=False,
#cust_grid_points=[[0, 1, 26, 56, 105], [0, 1.,  1.6, 2.5]])
#plt.savefig(os.path.join(model.path, "Light_intensity_Catalyst_loading.png"), dpi=500)
#plt.show()

#prediction_dist_heatmap(summary_df,
#                        font_cmap=None,
#                        annot_fontsize=6,
#                        cmap="tab20",
#                        xlabel=None,
#                        ylabel=None,
#                        )
#plt.savefig(os.path.join(model.path, "Light_intensity_Catalyst_loading_heatmap.png"), dpi=500)
#plt.show()



# importances = model._model.feature_importances_
#
# imp_df = pd.DataFrame(importances.reshape(1, -1), columns=new_inputs)
# imp_df.to_csv(os.path.join(model.path, "feature_importance.csv"))
#
# anion_labels = [f'Anions_{i}' for i in range(6)]
# anion_imp = imp_df[anion_labels].sum(axis=1)
# cat_labels = [f'catalyst_{i}' for i in range(15)]
# cat_imp = imp_df[cat_labels].sum(axis=1)
#
# rem_labels = [label for label in new_inputs if label not in anion_labels + cat_labels]
# rem_imp = imp_df[rem_labels]
#
# labels = ["Anion", "Catalyt_type"] + rem_labels
# vals = [float(anion_imp)] + [float(cat_imp)] + rem_imp.values.reshape(-1,).tolist()
#
# bar_chart(vals,
#           labels=labels)
#
# bar_chart(np.array(vals)/np.sum(vals), labels)