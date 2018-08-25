import sys

from core.cyclegan import CycleGAN
from core.models.models import *
from predictor import Predictor
from training.dataset import Dataset
from training.trainer import Trainer
from util.config import Config
from util.save_util import SaveUtil


def memory_hackermann():
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    c.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(tf.Session(config=c))


def predict(config: Config, cyclegan: CycleGAN, a_to_b=True):
    """
    Translate an entire folder from A -> B or vice versa.

    Input dir = config.predict_input_dir
    Output dir = config.predict_output_dir
    """

    f = cyclegan.a_to_b if a_to_b else cyclegan.b_to_a

    predictor = Predictor(f, config.image_shape)
    predictor.predict_directory(config.predict_input_dir, config.predict_output_dir)


def train(config: Config, cyclegan: CycleGAN):
    """
    Train CycleGAN using the hyperparameters in config.
    """

    # Initialize trainer
    trainer = Trainer(config, cyclegan, dataset)

    # Callback for saving checkpoints.
    def checkpoint(epoch):
        if epoch % config.checkpoint_interval == 0:
            save_util.save_checkpoint(epoch, cyclegan)

    # Callback for saving snapshots.
    def snapshot(iteration):
        if iteration % config.snapshot_interval == 0:
            save_util.save_snapshot(iteration, cyclegan)

    # Register callbacks
    trainer.on_epoch_start.append(checkpoint)
    trainer.on_iteration_start.append(snapshot)

    # Train the model
    trainer.train(n_epochs=config.n_epochs, batch_size=config.batch_size, start_epoch=epoch)


if __name__ == '__main__':
    # In case of weird out of memory error, try this:
    # memory_hackermann()

    config = Config()
    dataset = Dataset(config)
    save_util = SaveUtil(config, dataset)

    if config.load_checkpoint:
        gen_a, gen_b, dis_a, dis_b, epoch = save_util.load_models()
    else:
        dis_a = Discriminator.build_model(config)
        dis_b = Discriminator.build_model(config)
        gen_a = GeneratorResidual.build_model(config)
        gen_b = GeneratorResidual.build_model(config)
        epoch = 0

    cyclegan = CycleGAN(config, dis_a, dis_b, gen_a, gen_b)

    args = sys.argv
    if len(args) < 2:
        train(config, cyclegan)
    if args[1] == "train":
        train(config, cyclegan)
    if args[1] == "predict":
        predict(config, cyclegan)
