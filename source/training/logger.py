import csv
import datetime
import time

import tensorflow as tf
from keras import backend
from keras.callbacks import TensorBoard

from core.cyclegan import CycleGAN, Config


class Logger:
    def __init__(self, config: Config, cyclegan: CycleGAN):
        self.cyclegan = cyclegan
        self.generative_model = cyclegan.generative_model
        self.adversarial_model = cyclegan.discriminative_model

        self.use_cycle_loss = config.use_cycle_loss
        self.use_identity_loss = config.use_identity_loss
        self.log_dir = f"{config.dir_output}/{config.name_dataset}/logs/"

        logname = f"log_{time.time()}.csv"
        self.callback = TensorBoard(self.log_dir, write_graph=False)
        self.callback.set_model(self.generative_model)
        self.writer = csv.writer(open(f'{self.log_dir}/{logname}', "w", newline=''))
        self.start_time = datetime.datetime.now()
        self.init = False

    def log_iteration(self, iteration, epoch, batch_idx, n_batches, loss_d, loss_g):
        """
        Logs an iteration to both tensorboard and a .csv file.
        """

        elapsed = datetime.datetime.now() - self.start_time
        print(f"Epoch [{epoch}] [{batch_idx}/{n_batches}] \t| Elapsed {elapsed} \t| Loss (D) {loss_d[0]}\t| Loss (G) {loss_g[0]}")

        values = self.collect_values(loss_d, loss_g)
        self.log_to_tensorboard(iteration, values)
        self.log_to_csv(iteration, values)

    def collect_values(self, loss_d, loss_g):
        values = [('hyperparameters/learning rate (gen)', backend.get_value(self.cyclegan.generative_model.model.optimizer.lr)),
                  ('hyperparameters/learning rate (dis)', backend.get_value(self.cyclegan.discriminative_model.model.optimizer.lr)),
                  ('hyperparameters/lambda gan loss (gen)', backend.get_value(self.cyclegan.l_gan_g)),
                  ('hyperparameters/lambda gan loss (dis)', backend.get_value(self.cyclegan.l_gan_d))]

        if self.use_cycle_loss:
            values.append(('hyperparameters/lambda cycle loss', backend.get_value(self.cyclegan.l_cycle)))

        if self.use_identity_loss:
            values.append(('hyperparameters/lambda identity loss', backend.get_value(self.cyclegan.l_id)))

        values.extend(loss_g)
        values.extend(loss_d)
        return values

    def log_to_tensorboard(self, iteration, values):
        names = [item[0] for item in values]
        data = [item[1] for item in values]

        for name, value in zip(names, data):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.callback.writer.add_summary(summary, iteration)
            self.callback.writer.flush()

    def log_to_csv(self, iteration, values):
        data = [item[1] for item in values]
        data.insert(0, iteration)

        if not self.init:
            names = [item[0] for item in values]
            names.insert(0, 'iteration')
            self.init = True
            self.writer.writerow(names)

        self.writer.writerow(data)
