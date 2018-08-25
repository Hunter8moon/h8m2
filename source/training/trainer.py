from core.cyclegan import CycleGAN, AdversarialLoss
from training.dataset import Dataset
from training.image_buffer import ImageBuffer
from training.logger import Logger
from util.config import Config


class Trainer:
    def __init__(self, config: Config, cyclegan: CycleGAN, dataset: Dataset):
        self.on_epoch_start = [cyclegan.update_hyperparameters]
        self.on_iteration_start = []
        self.cyclegan = cyclegan
        self.dataset = dataset
        self.logger = Logger(config, cyclegan)

        self.pool_size = config.pool_size
        self.use_wasserstein_loss = config.adversarial_loss == AdversarialLoss.WGAN
        self.n_critic = config.n_critic

    def train(self, n_epochs=200, batch_size=1, start_epoch=0):
        n_epochs = n_epochs if n_epochs > 0 else 10_000_000

        print("Training models . . .")
        buffer_a = ImageBuffer(buffer_size=self.pool_size, batch_size=batch_size)
        buffer_b = ImageBuffer(buffer_size=self.pool_size, batch_size=batch_size)
        it = 0
        for epoch in range(start_epoch, n_epochs):
            self.epoch_start(epoch)
            batches = self.dataset.partition_files(batch_size)
            batch_idx = 0

            for a, b in batches:
                # The final batch may have different size, we just skip those.
                if len(a) != len(b) or len(a) != batch_size:
                    continue

                self.interation_start(it)

                # Load the batch of images.
                batch_a, batch_b = self.dataset.load_image_batch(a, b)

                # Train discriminators
                loss_d = self.cyclegan.train_discriminators(batch_a, batch_b, buffer_a, buffer_b)

                # For wgan, train for more iterations.
                if self.use_wasserstein_loss:
                    for i in range(self.n_critic - 1):
                        loss_d = self.cyclegan.train_discriminators(batch_a, batch_b, buffer_a, buffer_b)

                # Train generators
                loss_g = self.cyclegan.train_generators(batch_a, batch_b)

                self.logger.log_iteration(it, epoch, batch_idx, len(batches), loss_d, loss_g)
                batch_idx += 1
                it += 1

        # Invoke callbacks a final time.
        self.interation_start(it)
        self.epoch_start(n_epochs)

    def interation_start(self, it):
        for handler in self.on_iteration_start:
            handler(it)

    def epoch_start(self, epoch):
        for handler in self.on_epoch_start:
            handler(epoch)
