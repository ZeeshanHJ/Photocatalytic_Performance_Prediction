import ai4water
print(ai4water.__version__)

import numpy as np
np.set_printoptions(suppress=True, linewidth=150)
import pandas as pd
from easy_mpl import plot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ai4water.utils.utils import prepare_data

from ai4water import Model

fpath = "grouped data (Pd-BFO) samples.xlsx"

df = pd.read_excel(fpath, header=4)

df = df.iloc[:, 0:-2]

cat_type_encoder = LabelEncoder()
df["Catalyst_type"] = cat_type_encoder.fit_transform(df["Catalyst_type"])

pl_type_encoder = LabelEncoder()
df["Pollutant_Type"] = pl_type_encoder.fit_transform(df["Pollutant_Type"])

anions_type_encoder = LabelEncoder()
df["Anions"] = anions_type_encoder.fit_transform(df["Anions"])

inputs =['Catalyst_type', 'Pollutant_Type', 'Catalyst_loading',
       'Light_intensity', 'time', 'solution_pH', 'HA', 'Anions', 'Ci']
lookback = 10  # experiment length
x = df[inputs].values.reshape(162, lookback, len(inputs))

y = df['Cf'].values.reshape(162, lookback)

train_x, val_x, train_y, val_y = train_test_split(x,y, test_size=0.3, random_state=313)

# # model building
layers = {
    "Input": {"shape": (lookback, len(inputs))},
    "LSTM": {"units": 32, "return_sequences": True},
    "LSTM_1": {"units": 1, "return_sequences": True},
    "Flatten": {}
}

model = Model(
    model={"layers": layers},
    train_fraction=1.0,
    val_fraction=0.3,
    split_random=True,
    input_features=inputs,
    output_features=['Cf'],
    epochs=500
)


# # model Training
h = model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y))
#
# # model prediction
p = model.predict(x=train_x, y=train_y)
# p = model.predict(data="validation")
