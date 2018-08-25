from keras import Input

from core.losses.cyclegan_loss import *
from training.image_buffer import ImageBuffer
from util.config import Config


class DiscriminativeModel:
    """
    Combines two discriminators into a single discriminative model
    """

    def __init__(self, config: Config, loss_terms: [LossTerm], dis_a: Model, dis_b: Model, gen_a: Model, gen_b: Model):
        self.model: Model = None
        self.config = config
        self.batch_size = config.batch_size
        self.dis_a = dis_a
        self.dis_b = dis_b
        self.gen_a = gen_a
        self.gen_b = gen_b

        self.loss_terms = loss_terms
        self.loss_names = ['discriminative_model/']
        for term in self.loss_terms:
            for name in term.names:
                self.loss_names.append(f'discriminative_model/{name}')

    def compile(self):
        self.dis_a.trainable = True
        self.dis_b.trainable = True
        outputs, losses, weights = [], [], []

        real_a = Input(shape=self.config.image_shape)
        fake_a = Input(shape=self.config.image_shape)
        real_b = Input(shape=self.config.image_shape)
        fake_b = Input(shape=self.config.image_shape)

        inputs = [real_a, fake_a, real_b, fake_b]
        for term in self.loss_terms:
            term.compile(self.dis_a, self.dis_b, self.gen_a, self.gen_b, real_a, fake_a, real_b, fake_b)
            outputs.extend(term.outputs)
            losses.extend(term.losses)
            weights.extend(term.weights)

        model = Model(inputs, outputs)
        model.compile(optimizer=self.config.optimizer_d,
                      loss=losses,
                      loss_weights=weights)

        self.model = model

    def train_on_batch(self, real_a, real_b, buffer_a: ImageBuffer = None, buffer_b: ImageBuffer = None):
        self.dis_a.trainable = True
        self.dis_b.trainable = True

        # Forward pass
        fake_b = self.gen_a.predict(real_a)
        fake_a = self.gen_b.predict(real_b)

        # Add images to buffer
        if buffer_a:
            buffer_a.add(fake_a)
            fake_a = buffer_a.get_batch()
        if buffer_b:
            buffer_b.add(fake_b)
            fake_b = buffer_b.get_batch()

        # Update discriminators
        inputs = [real_a, fake_a, real_b, fake_b]
        outputs = []
        for term in self.loss_terms:
            outputs.extend(term.get_targets(self.batch_size, self.dis_b, self.gen_b))

        d_loss = self.model.train_on_batch(inputs, outputs)

        return list(zip(self.loss_names, d_loss))


class GenerativeModel:
    """
    Combines two generators into a single generative model
    """

    def __init__(self, config: Config, loss_terms: [LossTerm], dis_a: Model, dis_b: Model, gen_a: Model, gen_b: Model):
        self.model: Model = None
        self.config = config
        self.batch_size = config.batch_size
        self.dis_a = dis_a
        self.dis_b = dis_b
        self.gen_a = gen_a
        self.gen_b = gen_b

        self.loss_terms = loss_terms
        self.loss_names = ['generative_model/']
        for term in self.loss_terms:
            for name in term.names:
                self.loss_names.append(f'generative_model/{name}')

    def compile(self):
        self.dis_a.trainable = False
        self.dis_b.trainable = False
        outputs, losses, weights = [], [], []

        real_a = Input(shape=self.config.image_shape)
        real_b = Input(shape=self.config.image_shape)

        # Forward pass
        fake_b = self.gen_a(real_a)
        fake_a = self.gen_b(real_b)

        inputs = [real_a, real_b]
        for term in self.loss_terms:
            term.compile(self.dis_a, self.dis_b, self.gen_a, self.gen_b, real_a, fake_a, real_b, fake_b)
            outputs.extend(term.outputs)
            losses.extend(term.losses)
            weights.extend(term.weights)

        model = Model(inputs, outputs)
        model.compile(optimizer=self.config.optimizer_g,
                      loss=losses,
                      loss_weights=weights)
        self.model = model

    def train_on_batch(self, real_a, real_b):
        self.dis_a.trainable = False
        self.dis_b.trainable = False

        # Update generators
        inputs = [real_a, real_b]
        outputs = []
        for term in self.loss_terms:
            outputs.extend(term.get_targets(self.batch_size, self.dis_b, self.gen_b))

        g_loss = self.model.train_on_batch(inputs, outputs)
        return list(zip(self.loss_names, g_loss))
