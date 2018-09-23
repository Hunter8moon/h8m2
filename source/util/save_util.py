import math
import os
import re
from glob import glob

from keras.models import load_model
from keras_contrib.layers import InstanceNormalization

from core.cyclegan import CycleGAN
from core.reflection_padding import ReflectionPadding2D
from core.spectral_normalization import *
from util.image_util import ImageUtil


def save_checkpoint(current_epoch: int, cyclegan: CycleGAN, n_epochs: int, dir_checkpoints: str):
    """
    Saves checkpoints of the models.
    """

    if not os.path.exists(dir_checkpoints):
        os.makedirs(dir_checkpoints)

    width = int(math.log10(n_epochs)) + 1
    n = str(current_epoch).zfill(width)
    cyclegan.gen_a.save(f"{dir_checkpoints}/{n}_gen_a.h5")
    cyclegan.gen_b.save(f"{dir_checkpoints}/{n}_gen_b.h5")
    cyclegan.dis_a.save(f"{dir_checkpoints}/{n}_dis_a.h5")
    cyclegan.dis_b.save(f"{dir_checkpoints}/{n}_dis_b.h5")


def save_snapshot(epoch: int, batch: int, cyclegan: CycleGAN, test_batch, image_shape: (int, int, int), dir_snapshots: str):
    """
    Saves a snapshot of the current iteration to {self.snapshot_dir}/{self.dataset.name}.
    Snapshot shows;
    on the top row:    real_a, gen_a(real_a), gen_b(gen_a(real_a)),
    on the bottom row: real_b, gen_b(real_b), gen_a(gen_b(real_b)).
    """

    def create_snapshot():
        a, b = [real_a], [real_b]
        fake_b = cyclegan.a_to_b(real_a)
        fake_a = cyclegan.b_to_a(real_b)
        a.append(fake_b)
        b.append(fake_a)

        if cyclegan.config.use_cycle_loss:
            cycle_a = cyclegan.b_to_a(fake_b)
            cycle_b = cyclegan.a_to_b(fake_a)
            a.append(cycle_a)
            b.append(cycle_b)

        if cyclegan.config.use_identity_loss:
            id_a = cyclegan.b_to_a(real_a)
            id_b = cyclegan.a_to_b(real_b)
            a.append(id_a)
            b.append(id_b)

        return [a, b]

    if not os.path.exists(dir_snapshots):
        os.makedirs(dir_snapshots)

    real_a, real_b = test_batch
    outputs = create_snapshot()

    img = ImageUtil.make_snapshot_image(outputs, image_shape[0], image_shape[1])
    ImageUtil.save(img, dir_snapshots, f"{epoch:04d}_{batch:06d}")


def load_single_model(model_name, dir_checkpoints):
    """
    Loads a single model checkpoint from disk.

    :return: (model_name, epoch)
    """

    # Find all files in checkpoints dir.
    files = glob(f'{dir_checkpoints}\*')

    # Filter by model_name
    files = list(filter(lambda x: model_name in x, files))

    # Sort by epochs and find the last one.
    list.sort(files)
    if files:
        model_filename = files[-1]
        _, file = os.path.split(model_filename)

        epoch = re.findall(r'\d+', file)[0]
        epoch = int(epoch)
    else:
        raise Exception(f'Unable to load {model_name} model from \"{dir_checkpoints}\".')

    # Load the model
    model = load_model_checkpoint(model_filename)

    return model, epoch


def load_model_checkpoint(model_filename):
    custom_objects = {'ReflectionPadding2D': ReflectionPadding2D,
                      "InstanceNormalization": InstanceNormalization,
                      "ConvSN2DTranspose": ConvSN2DTranspose,
                      "EmbeddingSN": EmbeddingSN,
                      "ConvSN3D": ConvSN3D,
                      "ConvSN1D": ConvSN1D,
                      "ConvSN2D": ConvSN2D,
                      "DenseSN": DenseSN,
                      }
    model = load_model(model_filename, custom_objects=custom_objects)
    return model


def load_models(dir_checkpoints):
    """
    Loads the gen_a, gen_b, dis_a, dis_b checkpoints from disk.

    :return: (gen_a, gen_b, dis_a, dis_b, epoch)
    """

    gen_a, epoch = load_single_model('gen_a', dir_checkpoints)
    gen_b, epoch = load_single_model('gen_b', dir_checkpoints)
    dis_a, epoch = load_single_model('dis_a', dir_checkpoints)
    dis_b, epoch = load_single_model('dis_b', dir_checkpoints)

    return gen_a, gen_b, dis_a, dis_b, epoch

# class SaveUtil:
#     def __init__(self, config: Config, snapshot_interval=100, checkpoint_interval=500):
#         self.snapshot_interval = snapshot_interval
#         self.checkpoint_interval = checkpoint_interval
#         self.config = config
#
#         self.name_dataset = config.dataset_name
#         self.dir_snapshots = f'{config.dir_output}/{self.name_dataset}/snapshots/'
#         self.dir_checkpoints = f'{config.dir_output}/{self.name_dataset}/checkpoints/'
#
#     def load_models(self):
#         """
#         Loads the model checkpoints from disk.
#
#         :return: (gen_a, gen_b, dis_a, dis_b, epoch)
#         """
#
#         gen_a = self.load_model_from_disk('gen_a')
#         gen_b = self.load_model_from_disk('gen_b')
#         dis_a = self.load_model_from_disk('dis_a')
#         dis_b = self.load_model_from_disk('dis_b')
#         epoch = self.get_epoch_num()
#
#         return gen_a, gen_b, dis_a, dis_b, epoch
#
#     def load_model_from_disk(self, model_name):
#         """
#         Loads a single model checkpoint from disk.
#         """
#
#         custom_objects = {'ReflectionPadding2D': ReflectionPadding2D}
#         gen_a, epoch = self.get_model_filename(model_name)
#         gen_a = load_model(gen_a, custom_objects=custom_objects)
#         return gen_a
#
#     def get_model_filename(self, name):
#         """
#         Returns the filename of the latest checkpoint for the model with given name.
#         :param name: name of the checkpoint.
#         :return: (filename, epoch)
#         """
#         files = glob(f'{self.dir_checkpoints}\*')
#         files = list(filter(lambda x: name in x, files))
#
#         # Sort by epochs and iterations and load the last one
#         list.sort(files)
#         if files:
#             model = files[-1]
#             _, file = os.path.split(model)
#
#             return model
#         else:
#             raise Exception(f'Unable to load {name} model from \"{self.dir_checkpoints}\".')
#
#     def get_epoch_num(self):
#         """
#         Returns the filename of the latest checkpoint for the model with given name.
#         :return: (filename, epoch)
#         """
#         files = glob(f'{self.dir_checkpoints}\*')
#         files = list(filter(lambda x: 'gen_a' in x, files))
#
#         # Sort by epochs and iterations and load the last one
#         list.sort(files)
#         if files:
#             model = files[-1]
#             _, file = os.path.split(model)
#             epoch = re.findall(r'\d+', file)[0]
#
#             return int(epoch)
