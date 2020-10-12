import tensorflow as tf
from rl755.models.common.sequential import SequentialModel

class LSTM(SequentialModel):
    def __init__(self, hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm = tf.keras.layers.LSTM(units=self.hidden_size, return_sequences=True)
        self.final_layer = tf.keras.layers.Dense(
            units=self.output_size, activation=None
        )

    def call(self, inputs, mask=None, training=None):
        x = self.lstm(inputs, training=training)
        self.hidden_output = x
        output = self.final_layer(x, training=training)
        return output

    def get_loss_fn(self):
        """Train using a MSE loss."""
        return tf.keras.losses.MeanSquaredError()

    def get_hidden_representation(self, x, mask=None, training=None, position=-1): 
        self(x, training=training, mask=mask)
        return self.hidden_output[..., position, :]