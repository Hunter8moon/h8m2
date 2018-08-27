from keras.layers import Lambda
from keras.layers.merge import _Merge

from core.losses.cyclegan_loss import *


class RandomWeightedAverage(_Merge):
    """
    Provides a (random) weighted average between real and generated image samples
    """

    def _merge_function(self, inputs):
        weights = K.random_uniform((1, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


class GradientPenalty(Layer):
    """
    From:
    https://arxiv.org/pdf/1704.00028.pdf
    """

    def __init__(self, gp_lambda, **kwargs):
        super(GradientPenalty, self).__init__(**kwargs)
        self.gp_lambda = gp_lambda

    def build(self, input_shapes):
        super(GradientPenalty, self).build(input_shapes)

    def call(self, inputs, **kwargs):
        y_pred = inputs[0]
        averaged_samples = inputs[1]
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = self.gp_lambda * K.square(gradient_l2_norm - 1)

        return gradient_penalty


class LipschitzPenalty(Layer):
    """
    From:
    https://arxiv.org/pdf/1709.08894.pdf
    """

    def __init__(self, **kwargs):
        super(LipschitzPenalty, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(LipschitzPenalty, self).build(input_shapes)

    def call(self, inputs, **kwargs):
        y_pred = inputs[0]
        averaged_samples = inputs[1]
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(K.maximum(0.0, gradient_l2_norm - 1))

        return gradient_penalty


class WGAN_Discriminator(LossTerm):
    """
    Wasserstein adversarial loss (for discriminator).
    """

    def __init__(self, weight=1, weight_gp=10, regulizer=LipschitzPenalty):
        super().__init__(names=['dis_a', 'gp_a',
                                'dis_b', 'gp_b'],
                         weights=[weight, weight_gp,
                                  weight, weight_gp])
        self.regulizer = regulizer

    def compile(self, dis_a, dis_b, gen_a, gen_b, real_a, fake_a, real_b, fake_b):
        def wgan(discriminator, real, fake):
            interpolation = RandomWeightedAverage()([real, fake])

            dis_real = discriminator(real)
            dis_fake = discriminator(fake)
            dis_int = discriminator(interpolation)
            gp = self.regulizer()([dis_int, interpolation])
            w_d = Lambda(lambda x: x[0] - x[1])([dis_real, dis_fake])
            return [w_d, gp]

        out = []
        out.extend(wgan(dis_a, real_a, fake_a))
        out.extend(wgan(dis_b, real_b, fake_b))

        self.outputs = out

    def get_targets(self, batch_size, dis, gen):
        if not self.targets:
            _ = np.ones(shape=shape_with_batch(batch_size, dis))
            self.targets = [_, _,
                            _, _]
        return self.targets


class WGAN_Generator(LossTerm):
    """
    Wasserstein adversarial loss (for generator).
    """

    def __init__(self, weight=1):
        super().__init__(names=['dis_a', 'dis_b'],
                         weights=[weight, weight])

    def compile(self, dis_a, dis_b, gen_a, gen_b, real_a, fake_a, real_b, fake_b):
        dis_fake_a = dis_a(fake_a)
        dis_fake_b = dis_b(fake_b)

        loss_fake_a = dis_fake_a
        loss_fake_b = dis_fake_b

        self.outputs = (loss_fake_a, loss_fake_b)

    def get_targets(self, batch_size, dis, gen):
        if not self.targets:
            _ = np.ones(shape=shape_with_batch(batch_size, dis))
            self.targets = [_, _]
        return self.targets
