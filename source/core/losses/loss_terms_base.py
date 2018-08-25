from enum import Enum

from keras import Model
from keras.engine import Layer
from keras.losses import *


class AdversarialLoss(Enum):
    LSGAN = 1
    RaLSGAN = 2
    WGAN = 3


def mean_loss(_, loss):
    return K.mean(loss)


def shape_with_batch(batch_size: int, model: Model):
    t = model.output_shape
    lst = list(t)
    lst[0] = batch_size
    s = tuple(lst)

    return s


class SquaredError(Layer):
    def __init__(self, label=None):
        super(SquaredError, self).__init__()
        self.label = label

    def build(self, input_shapes):
        super(SquaredError, self).build(input_shapes)

    def call(self, inputs, **kwargs):
        if self.label is None:
            out = inputs[0] - inputs[1]
        else:
            out = inputs - self.label

        return K.square(out)


class AbsoluteError(Layer):
    def __init__(self, label=None):
        super(AbsoluteError, self).__init__()
        self.label = label

    def build(self, input_shapes):
        super(AbsoluteError, self).build(input_shapes)

    def call(self, inputs, **kwargs):
        if self.label is None:
            out = inputs[0] - inputs[1]
        else:
            out = inputs - self.label

        return K.abs(out)


class LossTerm:
    """
    Base class for loss terms.
    """

    def __init__(self, names, weights=None, losses=None):
        if weights is None: weights = [1] * len(names)
        if losses is None: losses = [mean_loss] * len(names)

        self.names = names
        self.weights = weights
        self.losses = losses

        self.outputs = []
        self.targets = []

    def compile(self, dis_a, dis_b, gen_a, gen_b, real_a, fake_a, real_b, fake_b):
        raise NotImplementedError()

    def get_targets(self, batch_size, dis, gen):
        raise NotImplementedError()
