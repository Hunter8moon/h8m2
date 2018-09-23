from core.losses.cyclegan_loss import *


class RelativisticLoss(Layer):
    def __init__(self, swap_fs=False, **kwargs):
        super(RelativisticLoss, self).__init__(**kwargs)

        def f1(input_tensor):
            return K.mean(K.pow(input_tensor - 1, 2), axis=0)

        def f2(input_tensor):
            return K.mean(K.pow(input_tensor + 1, 2), axis=0)

        self.f1 = f1
        self.f2 = f2

        # Swap f1 and f2 for generator loss.
        if swap_fs:
            self.f1, self.f2 = self.f2, self.f1

    def call(self, inputs, **kwargs):
        pred_avg_real = K.mean(inputs[0], axis=0)
        pred_avg_fake = K.mean(inputs[1], axis=0)
        rel_avg_real = inputs[0] - pred_avg_fake
        rel_avg_fake = inputs[1] - pred_avg_real

        return (self.f1(rel_avg_real) + self.f2(rel_avg_fake)) / 2.


class RaLSGAN_Discriminator(LossTerm):
    """
    Relativistic average Least Squares GAN loss (for discriminator).
    """

    def __init__(self, weight=1):
        super().__init__(names=['dis_a', 'dis_b'],
                         weights=[weight, weight])

    def compile(self, dis_a, dis_b, gen_a, gen_b, real_a, fake_a, real_b, fake_b):
        pred_a_real = dis_a(real_a)
        pred_a_fake = dis_a(fake_a)
        pred_b_real = dis_b(real_b)
        pred_b_fake = dis_b(fake_b)

        loss_a = RelativisticLoss()([pred_a_real, pred_a_fake])
        loss_b = RelativisticLoss()([pred_b_real, pred_b_fake])

        self.outputs = [loss_a, loss_b]

    def get_targets(self, batch_size, dis, gen):
        if not self.targets:
            _ = np.ones(shape=shape_with_batch(batch_size, dis))
            self.targets = [_, _]
        return self.targets


class RaLSGAN_Generator(LossTerm):
    """
    Relativistic average Least Squares GAN loss (for generator).
    """

    def __init__(self, weight=1):
        super().__init__(names=['dis_a', 'dis_b'],
                         weights=[weight, weight])

    def compile(self, dis_a, dis_b, gen_a, gen_b, real_a, fake_a, real_b, fake_b):
        pred_a_real = dis_a(real_a)
        pred_a_fake = dis_a(fake_a)
        pred_b_real = dis_b(real_b)
        pred_b_fake = dis_b(fake_b)

        loss_a = RelativisticLoss(swap_fs=True)([pred_a_real, pred_a_fake])
        loss_b = RelativisticLoss(swap_fs=True)([pred_b_real, pred_b_fake])

        self.outputs = [loss_a, loss_b]

    def get_targets(self, batch_size, dis, gen):
        if not self.targets:
            _ = np.ones(shape=shape_with_batch(batch_size, dis))
            self.targets = [_, _]
        return self.targets
