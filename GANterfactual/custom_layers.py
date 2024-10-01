import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

    def get_config(self):
        # Include any custom argument (such as epsilon) in the config
        config = super(ReflectionPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config

class ForegroundLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(ForegroundLayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Create trainable gamma and beta parameters for scaling and shifting
        self.gamma = self.add_weight(shape=(input_shape[-1],), initializer="ones", trainable=True, name="gamma")
        self.beta = self.add_weight(shape=(input_shape[-1],), initializer="zeros", trainable=True, name="beta")

    def call(self, inputs, mask):

        mask = tf.cast(mask, dtype=tf.float32)

        # Compute the number of foreground elements per channel
        masked_count = tf.reduce_sum(mask, axis=[1, 2], keepdims=True)

        # Calculate masked mean and variance (foreground only)
        masked_mean = tf.reduce_sum(inputs * mask, axis=[1, 2], keepdims=True) / masked_count
        masked_variance = tf.reduce_sum(((inputs - masked_mean) * mask) ** 2, axis=[1, 2], keepdims=True) / masked_count

        # Normalize using the masked mean and variance
        norm_output = (inputs - masked_mean) / tf.sqrt(masked_variance + self.epsilon)

        # Apply gamma and beta parameters for scaling and shifting
        output = self.gamma * norm_output + self.beta

        return output

    def get_config(self):
        # Include any custom argument (such as epsilon) in the config
        config = super(ForegroundLayerNormalization, self).get_config()
        config.update({"epsilon": self.epsilon})
        return config
