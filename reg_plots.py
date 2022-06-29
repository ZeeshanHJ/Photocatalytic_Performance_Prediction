
import os

import pandas as pd

from easy_mpl import regplot, plot

fpath = r"D:\Zeeshan\modeling\results\Efficiency_ohe_0_optimize_20220512_114057\CatBoostRegressor\best\20220512_115620"

df = pd.read_csv(os.path.join(fpath, "total_data1.csv"))

# catalyst_type
cond1 = (df['Light_intensity'] == 105.0) & (df['Anions']=='without Anion')

cond2 = (df['Catalyst_loading']==1.0) & (df['solution_pH']==5.45) & (df['Ci']==10.0) & (df['HA']==0.0)

df_obj_cat_type = df.loc[cond1 & cond2]

regplot(df_obj_cat_type["true_Efficiency"], df_obj_cat_type["pred_Efficiency"],
        title="Catalyst Type")
print(df_obj_cat_type.shape, "Catalyst Type")
fname = os.path.join(fpath, "catalyst_type_rgr.csv")
df_obj_cat_type[["true_Efficiency", "pred_Efficiency"]].to_csv(fname)


# Light Intensity
cond1 = (df['Catalyst_type'] == '2 wt% Pd-BFO') & (df['Anions']=='without Anion')

cond2 = (df['Catalyst_loading']==1.0) & (df['solution_pH']==5.45) & (df['Ci']==10.0) & (df['HA']==0.0)

df_obj_light = df.loc[cond1 & cond2]

regplot(df_obj_light["true_Efficiency"], df_obj_light["pred_Efficiency"], title="Light")
print(df_obj_light.shape, "Light Intensity")
fname = os.path.join(fpath, "Light_Intensity_rgr.csv")
df_obj_light[["true_Efficiency", "pred_Efficiency"]].to_csv(fname)


# catalyst_loading
cond1 = (df['Catalyst_type'] == '2 wt% Pd-BFO') & (df['Anions']=='without Anion')

cond2 = (df['Light_intensity']==105.0) & (df['solution_pH']==5.45) & (df['Ci']==10.0) & (df['HA']==0.0)

df_obj_cat = df.loc[cond1 & cond2]

regplot(df_obj_cat["true_Efficiency"], df_obj_cat["pred_Efficiency"], title="Catalyst Loading")
print(df_obj_cat.shape, "catalyst_loading")
fname = os.path.join(fpath, "catalyst_loading_rgr.csv")
df_obj_cat[["true_Efficiency", "pred_Efficiency"]].to_csv(fname)


# Initial Concentration
cond1 = (df['Catalyst_type'] == '2 wt% Pd-BFO') & (df['Anions']=='without Anion')

cond2 = (df['Light_intensity']==105.0) & (df['solution_pH']==5.45) & (df['Catalyst_loading']==1.5) & (df['HA']==0.0)

df_obj_ci = df.loc[cond1 & cond2]

regplot(df_obj_ci["true_Efficiency"], df_obj_ci["pred_Efficiency"],
        title="Initial Pollutant Concentration")
print(df_obj_ci.shape, "Initial Concentration")
fname = os.path.join(fpath, "Initial_Concentration_rgr.csv")
df_obj_ci[["true_Efficiency", "pred_Efficiency"]].to_csv(fname)


# Solution pH
cond1 = (df['Catalyst_type'] == '2 wt% Pd-BFO') & (df['Anions']=='without Anion')

cond2 = (df['Light_intensity']==105.0) & (df['Ci']==5.0) & (df['Catalyst_loading']==1.5) & (df['HA']==0.0)

df_obj_ph = df.loc[cond1 & cond2]

regplot(df_obj_ph["true_Efficiency"], df_obj_ph["pred_Efficiency"],
        title="Solution pH")
print(df_obj_ph.shape, "solution pH")
fname = os.path.join(fpath, "solution pH_rgr.csv")
df_obj_ph[["true_Efficiency", "pred_Efficiency"]].to_csv(fname)


# humic acid
cond1 = (df['Catalyst_type'] == '2 wt% Pd-BFO') & (df['Anions']=='without Anion')

cond2 = (df['Light_intensity']==105.0) & (df['Ci']==5.0) & (df['Catalyst_loading']==1.5) & (df['solution_pH']==7.0)

df_obj_ha = df.loc[cond1 & cond2]

regplot(df_obj_ha["true_Efficiency"], df_obj_ha["pred_Efficiency"],
        title="Humid Acid")
print(df_obj_ha.shape, "Humid Acid")
fname = os.path.join(fpath, "Humid_Acid_rgr.csv")
df_obj_ha[["true_Efficiency", "pred_Efficiency"]].to_csv(fname)


# Anions
cond1 = (df['Catalyst_type'] == '2 wt% Pd-BFO') & (df['HA']==0.0)

cond2 = (df['Light_intensity']==105.0) & (df['Ci']==5.0) & (df['Catalyst_loading']==1.5) & (df['solution_pH']==7.0)

df_obj_anions = df.loc[cond1 & cond2]

regplot(df_obj_anions["true_Efficiency"], df_obj_anions["pred_Efficiency"],
        title="Anions")
print(df_obj_anions.shape, "Anions")
fname = os.path.join(fpath, "Anions_rgr.csv")
df_obj_anions[["true_Efficiency", "pred_Efficiency"]].to_csv(fname)