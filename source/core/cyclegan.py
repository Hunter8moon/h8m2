from core.losses.relativisitic_loss import *
from core.losses.wasserstein_loss import *
from core.models.combined_models import *


def get_shape(batch_size: int, model: Model):
    t = model.output_shape
    lst = list(t)
    lst[0] = batch_size
    s = tuple(lst)

    return s


class CycleGAN:
    def __init__(self,
                 config: Config,
                 discriminator_a: Model,
                 discriminator_b: Model,
                 generator_a: Model,
                 generator_b: Model):
        self.config = config
        self.schedules = []

        # Build individual model
        self.gen_a: Model = generator_a
        self.gen_b: Model = generator_b
        self.dis_a: Model = discriminator_a
        self.dis_b: Model = discriminator_b

        # Initialize loss terms
        self.l_id = K.variable(1)
        self.l_gan_d = K.variable(1)
        self.l_gan_g = K.variable(1)
        self.l_cycle = K.variable(1)
        d_loss_terms, g_loss_terms = self.initialize_loss_terms(config)

        # Combine adversarial and generative models
        self.discriminative_model = DiscriminativeModel(config,
                                                        loss_terms=d_loss_terms,
                                                        dis_a=self.dis_a,
                                                        dis_b=self.dis_b,
                                                        gen_a=self.gen_a,
                                                        gen_b=self.gen_b)
        self.generative_model = GenerativeModel(config,
                                                loss_terms=g_loss_terms,
                                                dis_a=self.dis_a,
                                                dis_b=self.dis_b,
                                                gen_a=self.gen_a,
                                                gen_b=self.gen_b)
        # Compile models
        self.discriminative_model.compile()
        self.generative_model.compile()

        # Bind variables to schedules
        self.bind_schedules(config)

    def initialize_loss_terms(self, config):
        """
        Returns two lists of loss terms for the discriminator and generator.
        """
        d_loss_terms = []
        g_loss_terms = []

        if config.adversarial_loss == AdversarialLoss.WGAN:
            d_loss_terms.append(WGAN_Discriminator(weight=self.l_gan_d))
            g_loss_terms.append(WGAN_Generator(weight=self.l_gan_g))

        if config.adversarial_loss == AdversarialLoss.RaLSGAN:
            d_loss_terms.append(RLSGAN_Discriminator(weight=self.l_gan_d))
            g_loss_terms.append(RLSGAN_Generator(weight=self.l_gan_g))

        if config.adversarial_loss == AdversarialLoss.LSGAN:
            d_loss_terms.append(LSGAN_Discriminator(weight=self.l_gan_d))
            g_loss_terms.append(LSGAN_Generator(weight=self.l_gan_g))

        if config.use_cycle_loss:
            g_loss_terms.append(CycleConsistencyLoss(weight=self.l_cycle))

        if config.use_identity_loss:
            g_loss_terms.append(IdentityLoss(weight=self.l_id))

        return d_loss_terms, g_loss_terms

    def bind_schedules(self, config):
        """
        Binds the hyperparameters to their respective schedules.
        """
        self.schedules.append(config.lr_g_schedule)
        config.lr_g_schedule.bind_to_variables([self.generative_model.model.optimizer.lr])

        self.schedules.append(config.lr_d_schedule)
        config.lr_d_schedule.bind_to_variables([self.discriminative_model.model.optimizer.lr])

        self.schedules.append(config.l_dis_schedule)
        config.l_dis_schedule.bind_to_variables([self.l_gan_g])

        if config.use_cycle_loss:
            self.schedules.append(config.l_cycle_schedule)
            config.l_cycle_schedule.bind_to_variables([self.l_cycle])

        if config.use_identity_loss:
            self.schedules.append(config.l_id_schedule)
            config.l_id_schedule.bind_to_variables([self.l_id])

    def train_discriminators(self, real_a, real_b, buffer_a: ImageBuffer = None, buffer_b: ImageBuffer = None):
        return self.discriminative_model.train_on_batch(real_a, real_b, buffer_a, buffer_b)

    def train_generators(self, real_a, real_b):
        return self.generative_model.train_on_batch(real_a, real_b)

    def b_to_a(self, image):
        return self.gen_b.predict(image)

    def a_to_b(self, image):
        return self.gen_a.predict(image)

    def update_hyperparameters(self, epoch):
        for s in self.schedules:
            s.update(epoch)

    def cycle(self, real_a, real_b):
        fake_b = self.a_to_b(real_a)
        cycle_a = self.b_to_a(fake_b)

        fake_a = self.b_to_a(real_b)
        cycle_b = self.a_to_b(fake_a)

        aba = [real_a, fake_b, cycle_a]
        bab = [real_b, fake_a, cycle_b]
        return [aba, bab]
