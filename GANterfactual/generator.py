from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers  import Input, Concatenate
from tensorflow.keras.layers  import LeakyReLU, MaxPooling2D
from tensorflow.keras.layers  import UpSampling2D, Conv2D, Lambda
from tensorflow.keras  import Model
import tensorflow as tf

from GANterfactual.custom_layers import ForegroundLayerNormalization, ReflectionPadding2D


def build_generator(img_shape, gf, channels):
    """U-Net Generator"""

    def conv2d(layer_input, mask, filters, f_size=3, name_prefix=""):
        """Layers used during downsampling"""
        d = ReflectionPadding2D(padding=(1, 1))(layer_input)
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='valid',
        use_bias=False, name=name_prefix + "_conv2d")(d)
        d = ForegroundLayerNormalization(name=name_prefix + "_foreground_ln")(d, mask)
        d = LeakyReLU(alpha=0.2, name=name_prefix + "_leakyrelu")(d)
        return d

    def deconv2d(layer_input, skip_input, mask, filters, f_size=3, dropout_rate=0, name_prefix=""):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2, name=name_prefix + "_upsampling")(layer_input)
        u = ReflectionPadding2D(padding=(1, 1))(u)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='valid', activation='relu',
                   use_bias=False, name=name_prefix + "_conv2d")(u)
        if dropout_rate:
            u = Dropout(dropout_rate, name=name_prefix + "_dropout")(u)
        u = ForegroundLayerNormalization(name=name_prefix + "_foreground_ln")(u, mask)
        u = Concatenate(name=name_prefix + "_concat")([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape, name="input_image")
    mask0 = Lambda(lambda x: tf.cast(x > 0, dtype=tf.float32), name="mask0")(d0)

    # Downsampling
    mask1 = MaxPooling2D(pool_size=2, padding='same', name="mask1")(mask0)
    d1 = conv2d(d0, mask1, gf, name_prefix="down1")

    mask2 = MaxPooling2D(pool_size=2, padding='same', name="mask2")(mask1)
    d2 = conv2d(d1, mask2, gf * 2, name_prefix="down2")

    mask3 = MaxPooling2D(pool_size=2, padding='same', name="mask3")(mask2)
    d3 = conv2d(d2, mask3, gf * 4, name_prefix="down3")

    mask4 = MaxPooling2D(pool_size=2, padding='same', name="mask4")(mask3)
    d4 = conv2d(d3, mask4, gf * 8, name_prefix="down4")

    # Upsampling
    u1 = deconv2d(d4, d3, mask3, gf * 4, name_prefix="up1")
    u2 = deconv2d(u1, d2, mask2, gf * 2, name_prefix="up2")
    u3 = deconv2d(u2, d1, mask1, gf, name_prefix="up3")

    u4 = UpSampling2D(size=2, name="upsampling_final")(u3)
    u4 = ReflectionPadding2D(padding=(1, 1))(u4)
    output_img = Conv2D(channels, kernel_size=3, strides=1, activation='sigmoid', name="output_image")(u4) * mask0

    return Model(d0, output_img)

