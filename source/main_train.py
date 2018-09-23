from core.models.models import *
from training.dataset import Dataset
from training.trainer import Trainer
from util.save_util import *


def memory_hackermann():
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    c.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(tf.Session(config=c))


def train(config: Config, cyclegan: CycleGAN):
    """
    Train CycleGAN using the hyperparameters in config.
    """

    # Initialize trainer
    trainer = Trainer(config, cyclegan, dataset)

    # Callback for saving checkpoints.
    def checkpoint(epoch):
        if epoch % config.checkpoint_interval == 0:
            save_checkpoint(epoch, cyclegan, config.n_epochs, dir_checkpoints)

    # Callback for saving snapshots.
    def snapshot(epoch, batch):
        if batch % config.snapshot_interval == 0:
            save_snapshot(epoch, batch, cyclegan, dataset.batch_test(1), config.image_shape, dir_snapshots)

    # Register callbacks
    trainer.on_epoch_start.append(checkpoint)
    trainer.on_iteration_start.append(snapshot)

    # Train the model
    trainer.train(n_epochs=config.n_epochs, batch_size=config.batch_size, start_epoch=epoch)


if __name__ == '__main__':
    # In case of CUDNN_STATUS_INTERNAL_ERROR, try this:
    memory_hackermann()

    config = Config()
    dataset = Dataset(config)

    dir_snapshots = f'{config.dir_output}/{config.name_dataset}/snapshots/'
    dir_checkpoints = f'{config.dir_output}/{config.name_dataset}/checkpoints/'
    if config.load_checkpoint:
        gen_a, gen_b, dis_a, dis_b, epoch = load_models(dir_checkpoints)
    else:
        dis_a = Discriminator.build_model(config)
        dis_b = Discriminator.build_model(config)
        gen_a = GeneratorResidual.build_model(config)
        gen_b = GeneratorResidual.build_model(config)
        epoch = 0

    cyclegan = CycleGAN(config, dis_a, dis_b, gen_a, gen_b)

    train(config, cyclegan)
