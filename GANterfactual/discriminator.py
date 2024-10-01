import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Lambda, MaxPooling2D
from tensorflow.keras import Model

from GANterfactual.custom_layers import ForegroundLayerNormalization


def build_discriminator(img_shape, df):
    def d_layer(layer_input, mask, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = ForegroundLayerNormalization()(d, mask)
        return d

    img = Input(shape=img_shape)
    mask0 = Lambda(lambda x: tf.cast(x > 0, dtype=tf.float32), name="mask0")(img)

    # Downsampling
    mask1 = MaxPooling2D(pool_size=2, padding='same', name="mask1")(mask0)
    mask2 = MaxPooling2D(pool_size=2, padding='same', name="mask2")(mask1)
    mask3 = MaxPooling2D(pool_size=2, padding='same', name="mask3")(mask2)
    mask4 = MaxPooling2D(pool_size=2, padding='same', name="mask4")(mask3)
    d1 = d_layer(img, mask1, df, normalization=False)
    d2 = d_layer(d1, mask2, df * 2)
    d3 = d_layer(d2, mask3, df * 4)
    d4 = d_layer(d3, mask4, df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, validity)