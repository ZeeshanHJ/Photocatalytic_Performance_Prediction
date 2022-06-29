import site
site.addsitedir("D:\\AI4Water")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easy_mpl import imshow

from ai4water import Model

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class PrepareData(object):


    def __call__(
            self,
            inputs:list,
            target="k",
            transformation = "le",
            lookback=10,
            return_sequences=False,
    ):

        if transformation:
            assert transformation in ("le", "ohe", "embedding")

        inputs = inputs.copy()

        fpath = "MG dye degradation deta.xlsx"
        df = pd.read_excel(fpath)

        if transformation == "le":
            cat_encoder = LabelEncoder()
            df["Catalyst_type"] = cat_encoder.fit_transform(df["Catalyst_type"])

            anion_encoder = LabelEncoder()
            df["Anions"] = anion_encoder.fit_transform(df["Anions"])

        elif transformation == "embedding":
            from tensorflow.keras.preprocessing.text import one_hot
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            vocab_size = 50
            encoded_docs = [one_hot(d, vocab_size) for d in df["Catalyst_type"]]
            max_len = 5
            encoded_cat  = pad_sequences(encoded_docs, maxlen=max_len, padding='post')
            new_features = [f"catalyst_{i}" for i in range(encoded_cat.shape[-1])]
            inputs += new_features
            inputs.remove("Catalyst_type")
            df[new_features] = encoded_cat

            encoded_an = [one_hot(d, vocab_size) for d in df["Anions"]]
            max_len_anion = 2
            encoded_anion  = pad_sequences(encoded_an, maxlen=max_len_anion, padding='post')
            new_features = [f"Anions_{i}" for i in range(encoded_anion.shape[-1])]
            inputs += new_features
            inputs.remove("Anions")
            df[new_features] = encoded_anion

        elif transformation == "ohe":
            df, inputs, cat_encoder, anion_encoder = decode_with_ohe(df, inputs)

        self.inputs = inputs
        self.cat_encoder = cat_encoder
        self.anion_encoder = anion_encoder

        # first order
        k = np.log(df["Ci"] / df["Cf"]) / df["time"]
        df["k"] = k

        # k second order
        k_2nd = ((1/df["Cf"]) - (1/df["Ci"])) / df["time"]
        df["k_2nd"] = k_2nd

        df['time'] = df.pop('time')

        if lookback>0:
            num_samples = int(df.shape[0]/lookback)
            x = df[inputs].values.reshape(num_samples, lookback, len(inputs))

            if return_sequences:
                y = df[target].values.reshape(num_samples, lookback)
            else:
                y = df[target].values.reshape(num_samples, lookback)[:, -5]

            assert np.isinf(y).sum()==0, f"{np.isinf(y).sum()}"
            assert np.isnan(y).sum()==0, f"{np.isnan(y).sum()}"

            new_inputs = []
            for i in inputs:
                for j in range(lookback):
                    new_inputs.append(f"{i}_{j}")
        else:
            x = df[inputs].values
            y = df[target].values.reshape(-1, 1)

        return x, y


def decode_with_ohe(df, inputs):

    cat_encoder = OneHotEncoder(sparse=False)
    ohe_cat = cat_encoder.fit_transform(df["Catalyst_type"].values.reshape(-1, 1))
    new_features = [f"catalyst_{i}" for i in range(ohe_cat.shape[-1])]
    inputs += new_features
    inputs.remove("Catalyst_type")
    df[new_features] = ohe_cat

    anion_encoder = OneHotEncoder(sparse=False)
    ohe_cat = anion_encoder.fit_transform(df["Anions"].values.reshape(-1, 1))
    new_features = [f"Anions_{i}" for i in range(ohe_cat.shape[-1])]
    inputs += new_features
    inputs.remove("Anions")
    df[new_features] = ohe_cat

    return df, inputs, cat_encoder, anion_encoder

def prepare_data(inputs):

    fpath = "grouped data (Pd-BFO) samples.xlsx"

    df = pd.read_excel(fpath, header=4)

    df = df.iloc[:, 0:-2]

    cat_type_encoder = LabelEncoder()
    df["Catalyst_type"] = cat_type_encoder.fit_transform(df["Catalyst_type"])

    pl_type_encoder = LabelEncoder()
    df["Pollutant_Type"] = pl_type_encoder.fit_transform(df["Pollutant_Type"])

    anions_type_encoder = LabelEncoder()
    df["Anions"] = anions_type_encoder.fit_transform(df["Anions"])

    k = np.log(df["Ci"]/df["Cf"])/df["time"]
    df["k"] = k

    lookback = 10  # experiment length
    x = df[inputs].values.reshape(162, lookback, len(inputs))

    y = df['k'].values.reshape(162, lookback)[:, -5]
    np.isinf(y).sum()

    new_inputs = []
    for i in inputs:
        for j in range(10):
            new_inputs.append(f"{i}_{j}")

    return x, y


class MyModel(Model):

    def fit_ml_models(self, inputs, outputs, **kwargs):

        inputs = inputs.reshape(len(inputs), -1)
        history = self._model.fit(inputs, outputs, **kwargs)

        if self._model.__class__.__name__.startswith("XGB") and inputs.__class__.__name__ == "ndarray":
            # by default feature_names of booster as set to f0, f1,...
            self._model.get_booster().feature_names = self.input_features

        self._save_ml_model()

        return history

    def predict_ml_models(self, inputs, **kwargs):
        """So that it can be overwritten easily for ML models."""

        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.values

        inputs = inputs.reshape(len(inputs), -1)
        return self.predict_fn(inputs, **kwargs)


def prediction_dist_heatmap(summary_df, annot_fontsize=8,
                            cmap="summer",
                            font_cmap=None,
                            xlabel=None,
                            ylabel=None,
                            yticklabels=None,
                            ):



    if font_cmap is None:
        font_color="black"
    else:
        font_color = plt.get_cmap("Blues")(1.)

    ann_kws = {"color": "w", "ha": "center", "va": "center",
               'fmt': '%.1f',
               'fontdict': {'size': annot_fontsize,
                            'color': font_color}
               }

    vals = []
    for i in np.unique(summary_df['x1']):
        row = summary_df['actual_prediction_q2'].loc[summary_df['x1'] == i]
        for j in np.unique(summary_df['x2'])[::-1]:
            vals.append(row.iloc[j])

    xticklabels = summary_df.loc[summary_df['x1'] == 0]['display_column_2'].values[::-1]

    if yticklabels is None:
        yticklabels = summary_df.loc[summary_df['x2'] == 0]['display_column_1'].values

    plt.close('all')

    x = np.array(vals).reshape(len(yticklabels), len(xticklabels))
    ax, _ = imshow(x,
                   colorbar=True,
                   annotate=True,
                   annotate_kws=ann_kws,
                   cmap=cmap,
                   aspect="auto",
                   yticklabels=yticklabels,
                   xlabel=xlabel,
                   ylabel=ylabel,
                   show=False
                   )

    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=90)
    plt.tight_layout()

    df = pd.DataFrame(x, columns=xticklabels, index=yticklabels)

    return df,  ax