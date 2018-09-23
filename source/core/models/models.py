from keras.layers import *
from keras.models import Model

from core.layer_builder import LayerBuilder
from core.reflection_padding import ReflectionPadding2D
from util.config import Config


class Discriminator:
    """
    https://arxiv.org/pdf/1703.10593.pdf


    For discriminator networks, we use 70×70 PatchGAN [21].
    Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2.
    After the last layer, we apply a convolution to produce a 1 dimensional output.
    We do not use InstanceNorm for the first C64 layer. We use leaky ReLUs with slope 0.2.
    The discriminator core is: C64-C128-C256-C512
    """

    @staticmethod
    def build_model(config: Config):
        lb = LayerBuilder(config, use_spectral_norm=config.use_spectral_normalization)
        lb.kernel_size = 4
        filters = config.dis_filters

        input = Input(shape=config.image_shape)

        c1 = lb.convolution(input, filters, activation=LeakyReLU(0.2), normalize=False)  # C64
        c2 = lb.convolution(c1, filters * 2, activation=LeakyReLU(0.2))  # C128
        c3 = lb.convolution(c2, filters * 4, activation=LeakyReLU(0.2))  # C256
        c4 = lb.convolution(c3, filters * 8, strides=1, activation=LeakyReLU(0.2))  # C512

        output = lb.convolution(c4, 1, strides=1, activation=None, normalize=False)

        model = Model(input, output)
        return model




class GeneratorUNet:
    @staticmethod
    def build_model(config: Config):
        lb = LayerBuilder(config)
        lb.kernel_size = 4

        base_filters = config.gen_filters
        max_filters = int(base_filters * 8)
        n = int(np.log2(config.image_shape[0]))

        input = Input(shape=config.image_shape)

        # Encoder
        convolutions = [input]
        for i in range(n):
            # First layers is not normalized
            normalize = False if i == 0 else True

            filters = min(base_filters * (2 ** i), max_filters)

            prev = convolutions[-1]
            conv = lb.convolution(prev, filters, activation=LeakyReLU(0.2), normalize=normalize)
            convolutions.append(conv)

        # Decoder
        deconvolutions = [convolutions[-1]]
        for i in range(n):
            # Skip idx
            j = n - i - 1
            skip = convolutions[j]

            # First 3 layers have dropout
            dropout = True if i <= 2 else False

            # Calculate # filters
            filters = int(min(base_filters * (2 ** (j - 1)), max_filters))

            prev = deconvolutions[-1]
            deconv = lb.convolution_transpose_skip(prev, skip, filters, dropout=dropout)
            deconvolutions.append(deconv)

        output = lb.convolution(deconvolutions[-1], config.channels, strides=1, activation='tanh', normalize=False)

        model = Model(input, output)
        return model


class GeneratorResidual:
    """
    Let c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1.
    dk denotes a 3×3 Convolution-InstanceNorm-ReLU layer with k filters, and stride 2. Reflection padding was used to reduce artifacts.
    Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer.
    uk denotes a 3×3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters, and stride 1/2.
    The network with 6 blocks consists of: c7s1-32,d64,d128,R128,R128,R128,R128,R128,R128,u64,u32,c7s1-3
    """

    @staticmethod
    def build_model(config: Config):
        lb = LayerBuilder(config)
        lb.kernel_size = 3

        padding_method = ReflectionPadding2D
        filters = config.gen_filters
        padding = ((1, 1), (1, 1))
        outer_kernel_size = 7
        outer_padding = ((3, 3), (3, 3))

        input = Input(shape=config.image_shape)

        output = input
        output = padding_method(outer_padding)(output)

        output = lb.convolution(output, filters, kernel_size=outer_kernel_size, strides=1, activation='linear', padding='valid')
        output = lb.convolution(output, filters * 2, strides=2)
        output = lb.convolution(output, filters * 4, strides=2)

        for i in range(config.residual_blocks):
            output = lb.residual_padded(output, filters * 4, padding=padding)

        output = lb.convolution_transpose(output, filters * 2, strides=2)
        output = lb.convolution_transpose(output, filters, strides=2)
        output = padding_method(outer_padding)(output)

        output = lb.convolution(output, config.channels, kernel_size=outer_kernel_size, strides=1, padding='valid', activation='tanh', normalize=False)

        model = Model(input, output)
        return model


class GeneratorResidualSimple:
    """
    Residual generator without fancy reflection padding and 4 × 4 kernels for every layer.
    """

    @staticmethod
    def build_model(config: Config):
        lb = LayerBuilder(config)
        lb.kernel_size = 4
        filters = config.gen_filters

        input = Input(shape=config.image_shape)
        output = input

        # Encoder
        output = lb.convolution(output, filters, strides=1, activation='linear')
        output = lb.convolution(output, filters * 2, strides=2)
        output = lb.convolution(output, filters * 4, strides=2)

        # Residual blocks
        for i in range(config.residual_blocks):
            output = lb.residual_block(output, filters * 4)

        # Decoder
        output = lb.convolution_transpose(output, filters * 2, strides=2)
        output = lb.convolution_transpose(output, filters, strides=2)
        output = lb.convolution(output, config.channels, strides=1, activation='tanh', normalize=False)

        model = Model(input, output)
        return model
