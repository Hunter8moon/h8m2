import numpy as np

from core.losses.loss_terms_base import *


class CycleConsistencyLoss(LossTerm):
    """
    Mean absolute error cycle consistency loss.
    """

    def __init__(self, weight=1):
        super().__init__(names=['cycle_a', 'cycle_b'],
                         weights=[weight, weight])

    def compile(self, dis_a, dis_b, gen_a, gen_b, real_a, fake_a, real_b, fake_b):
        cycle_a = gen_b(fake_b)
        cycle_b = gen_a(fake_a)

        cycle_loss_a = AbsoluteError()([real_a, cycle_a])
        cycle_loss_b = AbsoluteError()([real_b, cycle_b])

        self.outputs = [cycle_loss_a, cycle_loss_b]

    def get_targets(self, batch_size, dis: Model, gen: Model):
        if not self.targets:
            _ = np.ones(shape=shape_with_batch(batch_size, gen))
            self.targets = [_, _]
        return self.targets


class IdentityLoss(LossTerm):
    """
    Mean absolute error identity loss.
    """

    def __init__(self, weight=1):
        super().__init__(names=['id_a', 'id_b'],
                         weights=[weight, weight])

    def compile(self, dis_a, dis_b, gen_a, gen_b, real_a, fake_a, real_b, fake_b):
        id_a = gen_b(real_a)
        id_b = gen_a(real_b)

        id_loss_a = AbsoluteError()([real_a, id_a])
        id_loss_b = AbsoluteError()([real_b, id_b])

        self.outputs = [id_loss_a, id_loss_b]

    def get_targets(self, batch_size, dis: Model, gen: Model):
        if not self.targets:
            _ = np.ones(shape=shape_with_batch(batch_size, gen))
            self.targets = [_, _]
        return self.targets


class LSGAN_Discriminator(LossTerm):
    """
    Least squares GAN adversarial loss (for discriminator).
    """

    def __init__(self, weight=1):
        super().__init__(names=['dis_real_a', 'dis_fake_a',
                                'dis_real_b', 'dis_fake_b'],
                         weights=[weight, weight, weight, weight])

    def compile(self, dis_a, dis_b, gen_a, gen_b, real_a, fake_a, real_b, fake_b):
        dis_real_a = dis_a(real_a)
        dis_fake_a = dis_a(fake_a)
        dis_real_b = dis_b(real_b)
        dis_fake_b = dis_b(fake_b)

        loss_real_a = SquaredError(1)(dis_real_a)
        loss_fake_a = SquaredError(0)(dis_fake_a)
        loss_real_b = SquaredError(1)(dis_real_b)
        loss_fake_b = SquaredError(0)(dis_fake_b)

        self.outputs = [loss_real_a, loss_fake_a, loss_real_b, loss_fake_b]

    def get_targets(self, batch_size, dis: Model, gen: Model):
        if not self.targets:
            _ = np.ones(shape=shape_with_batch(batch_size, dis))
            self.targets = [_, _, _, _]
        return self.targets


class LSGAN_Generator(LossTerm):
    """
    Least squares GAN adversarial loss (for generator).
    """

    def __init__(self, weight=1):
        super().__init__(names=['dis_a', 'dis_b'],
                         weights=[weight, weight])

    def compile(self, dis_a, dis_b, gen_a, gen_b, real_a, fake_a, real_b, fake_b):
        dis_fake_a = dis_a(fake_a)
        dis_fake_b = dis_b(fake_b)

        loss_fake_a = SquaredError(1)(dis_fake_a)
        loss_fake_b = SquaredError(1)(dis_fake_b)

        self.outputs = [loss_fake_a, loss_fake_b]

    def get_targets(self, batch_size, dis: Model, gen: Model):
        if not self.targets:
            _ = np.ones(shape=shape_with_batch(batch_size, dis))
            self.targets = [_, _]
        return self.targets
