import site
site.addsitedir("D:\\AI4Water")

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np

from tensorflow.keras.models import Model as KModel
from tensorflow.keras.layers import Layer, Dense, Activation, Lambda, LSTM, Input, Concatenate
from ai4water.models.tensorflow import EALSTM


class LightLSTM(EALSTM):

    def call(self, inputs, static_inputs, initial_state=None, **kwargs):
        """
        static_inputs :
            of shape (batch, num_static_inputs)
        """
        if not self.time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])
            #static_inputs = tf.transpose(static_inputs, [1, 0, 2])

        lookback, batch_size, _ = inputs.shape

        if initial_state is None:
            initial_state = tf.zeros((batch_size, self.units))  # todo
            state = [initial_state, initial_state]
        else:
            state = initial_state

        static_inputs_last = static_inputs[:, -1]

        # calculate input gate only once because inputs are static
        inp_g = self.input_gate(static_inputs_last)  # (batch, num_static_inputs) -> (batch, units)

        outputs, states = [], []
        for time_step in range(lookback):

            _out, state = self.cell(inputs[time_step], inp_g, state)

            outputs.append(_out)
            states.append(state)

        outputs = tf.stack(outputs)
        h_s = tf.stack([states[i][0] for i in range(lookback)])
        c_s = tf.stack([states[i][1] for i in range(lookback)])

        if not self.time_major:
            outputs = tf.transpose(outputs, [1, 0, 2])
            h_s = tf.transpose(h_s, [1, 0, 2])
            c_s = tf.transpose(c_s, [1, 0, 2])
            states = [h_s, c_s]
            last_output = outputs[:, -1]
        else:
            states = [h_s, c_s]
            last_output = outputs[-1]

        h = last_output

        if self.return_sequences:
            h = outputs

        if self.return_state:
            return h, states

        return h



if __name__ == "__main__":
    batch_size, lookback, num_dyn_inputs, num_static_inputs, units = 10, 5, 1, 10, 16

    inp_dyn = Input(batch_shape=(batch_size, lookback, num_dyn_inputs))
    inp_static = Input(batch_shape=(batch_size, lookback, num_static_inputs))


    lstm = LightLSTM(units, num_static_inputs)(inp_dyn, inp_static)

    # new_input = Concatenate()([inp_dyn, inp_static])
    # lstm = LSTM(units)(new_input)

    out = Dense(1)(lstm)
    model = KModel(inputs=[inp_dyn, inp_static], outputs=out)
    model.compile(loss='mse')
    model.summary()

    x1 = np.random.random((100, 5, 1))
    x2 = np.random.random((100, 5, 10))
    y = np.random.random((100, 1))
    model.fit(x=[x1, x2], y=y)