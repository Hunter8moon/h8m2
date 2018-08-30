from keras.initializers import *
from keras.layers import *
from keras_contrib.layers import InstanceNormalization

from core.reflection_padding import ReflectionPadding2D
from core.spectral_normalization import *
from util.config import Config


class LayerBuilder:
    def __init__(self, config: Config, use_spectral_norm=False):
        self.config = config
        self.kernel_size = 4
        self.strides = 2
        self.dropout_rate = config.dropout_rate
        self.use_resize_convolution = config.use_resize_convolution
        self.padding_method = ReflectionPadding2D

        self.init = RandomNormal(mean=0, stddev=0.02)
        self.norm = InstanceNormalization()

        if use_spectral_norm:
            self.dense = DenseSN
            self.conv2d = ConvSN2D
            self.conv2d_t = ConvSN2DTranspose
        else:
            self.dense = Dense
            self.conv2d = Conv2D
            self.conv2d_t = Conv2DTranspose

    def residual(self, input, filters, kernel_size=None):
        skip_layer = input

        output = self.convolution(input, filters, strides=1, kernel_size=kernel_size)
        output = self.convolution(output, filters, strides=1, kernel_size=kernel_size, activation=None)
        output = Add()([output, skip_layer])

        return output

    def residual_padded(self, input, filters, kernel_size=None, padding=None):
        skip_layer = input

        output = input
        if padding:
            output = self.padding_method(padding=((1, 1), (1, 1)))(output)
        output = self.convolution(output, filters, strides=1, kernel_size=kernel_size, padding='valid')
        if padding:
            output = self.padding_method(padding=((1, 1), (1, 1)))(output)
        output = self.convolution(output, filters, strides=1, kernel_size=kernel_size, padding='valid', activation=None)

        output = Add()([output, skip_layer])
        output = Activation(activation='relu')(output)
        return output

    def convolution_transpose_skip(self, input_layer, skip_layer, filters, kernel_size=None, strides=None, dropout_rate=None, dropout=False, normalize=True, activation='relu'):
        output = self.convolution_transpose(input_layer, filters, kernel_size, strides, dropout_rate, dropout, normalize, activation)
        output = Concatenate()([output, skip_layer])

        return output

    def convolution_transpose(self, input_layer, filters, kernel_size=None, strides=None, dropout_rate=None, dropout=False, normalize=True, activation='relu'):
        if not kernel_size:
            kernel_size = self.kernel_size
        if not strides:
            strides = self.strides
        if not dropout_rate:
            dropout_rate = self.dropout_rate

        if self.use_resize_convolution:
            output = UpSampling2D(strides)(input_layer)
            output = self.conv2d(filters=filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=self.init)(output)
        else:
            init = self.init
            output = self.conv2d_t(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=init)(input_layer)

        if normalize:
            output = self.norm(output)

        if activation:
            if isinstance(activation, str):
                output = Activation(activation=activation)(output)
            else:
                output = activation()(output)

        if dropout and dropout_rate > 0:
            output = Dropout(rate=dropout_rate)(output)

        return output

    def convolution(self, input_layer, filters, kernel_size=None, strides=None, dropout_rate=None, dropout=False, normalize=True, activation='relu', padding='same'):
        if not kernel_size:
            kernel_size = self.kernel_size
        if not strides:
            strides = self.strides
        if not dropout_rate:
            dropout_rate = self.dropout_rate

        output = self.conv2d(filters, use_bias=False, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=self.init)(input_layer)

        if normalize:
            output = self.norm(output)

        if activation:
            if isinstance(activation, str):
                output = Activation(activation=activation)(output)
            else:
                output = activation(output)

        if dropout and dropout_rate > 0:
            output = Dropout(rate=dropout_rate)(output)

        return output
