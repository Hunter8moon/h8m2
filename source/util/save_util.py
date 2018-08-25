import math
import os
import re
from glob import glob

from keras.models import load_model

from core.cyclegan import CycleGAN
from core.reflection_padding import ReflectionPadding2D
from training.dataset import Dataset
from util.config import Config
from util.image_util import ImageUtil


class SaveUtil:
    def __init__(self, config: Config, dataset: Dataset, snapshot_interval=100, checkpoint_interval=500):
        self.snapshot_interval = snapshot_interval
        self.checkpoint_interval = checkpoint_interval
        self.config = config
        self.dataset = dataset
        self.snapshot_dir = config.snapshot_dir
        self.checkpoints_dir = config.checkpoints_dir

    def save_checkpoint(self, epoch, cyclegan: CycleGAN):
        """
        Saves checkpoints of the models to {self.checkpoints_dir}/{self.dataset.name}.
        """

        directory = f'{self.checkpoints_dir}/{self.dataset.name}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        l = int(math.log10(self.config.n_epochs)) + 1
        n = str(epoch).zfill(l)
        cyclegan.gen_a.save(f"{directory}/{n}_{self.dataset.name}_gen_a.h5")
        cyclegan.gen_b.save(f"{directory}/{n}_{self.dataset.name}_gen_b.h5")
        cyclegan.dis_a.save(f"{directory}/{n}_{self.dataset.name}_dis_a.h5")
        cyclegan.dis_b.save(f"{directory}/{n}_{self.dataset.name}_dis_b.h5")

    def save_snapshot(self, iteration, cyclegan: CycleGAN):
        """
        Saves a snapshot of the current iteration to {self.snapshot_dir}/{self.dataset.name}.
        Snapshot shows;
        on the top row:    real_a, gen_a(real_a), gen_b(gen_a(real_a)),
        on the bottom row: real_b, gen_b(real_b), gen_a(gen_b(real_b)).

        :param iteration: The iteration number.
        :param cyclegan: The model.
        """

        real_a, real_b = self.dataset.batch_test(1)
        outputs = cyclegan.cycle(real_a, real_b)

        img = ImageUtil.make_snapshot_image(outputs, self.config.image_shape[0], self.config.image_shape[1])
        ImageUtil.save(img, f"{self.snapshot_dir}/{self.dataset.name}", f"{iteration:010d}")

    def load_models(self):
        """
        Loads the model checkpoints from disk.
        Assumes the checkpoints are in config.checkpoints_dir.

        :return: (gen_a, gen_b, dis_a, dis_b, epoch)
        """

        gen_a, epoch = self.get_model_filename('gen_a')
        gen_b, epoch = self.get_model_filename('gen_b')
        dis_a, epoch = self.get_model_filename('dis_a')
        dis_b, epoch = self.get_model_filename('dis_b')

        custom_objects = {'ReflectionPadding2D': ReflectionPadding2D}
        gen_a = load_model(gen_a, custom_objects=custom_objects)
        gen_b = load_model(gen_b, custom_objects=custom_objects)
        dis_a = load_model(dis_a, custom_objects=custom_objects)
        dis_b = load_model(dis_b, custom_objects=custom_objects)

        return gen_a, gen_b, dis_a, dis_b, epoch

    def get_model_filename(self, name):
        """
        Returns the filename of the latest checkpoint for the model with given name.
        :param name: name of the checkpoint.
        :return: (filename, epoch)
        """
        checkpoints_dir = self.config.checkpoints_dir
        dataset_name = self.config.dataset_name

        directory = f'{checkpoints_dir}\{dataset_name}'
        files = glob(f'{directory}\*')
        files = list(filter(lambda x: name in x, files))

        # Sort by epochs and iterations and load the last one
        list.sort(files)
        if files:
            model = files[-1]
            _, file = os.path.split(model)
            epoch = re.findall(r'\d+', file)[0]

            return model, int(epoch)
        else:
            raise Exception(f'Unable to load {name} model from \"{checkpoints_dir}\" for \"{dataset_name}\" ')
