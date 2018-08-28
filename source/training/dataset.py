import os
import random
from glob import glob

import numpy as np

from util.config import Config
from util.image_util import ImageUtil


class Dataset:
    def __init__(self, config: Config):
        self.dataset_path = os.path.join(config.dir_input, config.name_dataset)
        self.shape = config.image_shape

        # Load the files in each folder
        folder_train_a = os.path.join(self.dataset_path, config.folder_train_a)
        folder_train_b = os.path.join(self.dataset_path, config.folder_train_b)
        folder_test_a = os.path.join(self.dataset_path, config.folder_test_a)
        folder_test_b = os.path.join(self.dataset_path, config.folder_test_b)
        self.files_trainA = glob(f'{folder_train_a}/*')
        self.files_trainB = glob(f'{folder_train_b}/*')
        self.files_testA = glob(f'{folder_test_a}/*')
        self.files_testB = glob(f'{folder_test_b}/*')

        # Fail fast if loading batches fails
        if not self.files_trainA:
            raise Exception(f"Unable to load training images for A.\n\t {self.files_trainA} has no files.")
        if not self.files_trainB:
            raise Exception(f'Unable to load training images for B.\n\t {self.files_trainB} has no files.')
        if not self.files_testA:
            raise Exception(f'Unable to load test images for A.\n\t {self.files_testA} has no files.')
        if not self.files_testB:
            raise Exception(f'Unable to load test images for B.\n\t {self.files_testB} has no files.')

    def load_batch(self, files, batch_size, augment=True):
        """
        Returns an random sample of images of length batch_size from the filenames.
        """

        batch_size = min(batch_size, len(files))
        files = random.sample(files, batch_size)

        w = self.shape[0]
        h = self.shape[1]
        images = []
        for file in files:
            img = ImageUtil.file_to_array(file, w, h, augment=augment)
            images.append(img)

        images = np.array(images)
        return images

    def batch_test(self, batch_size):
        samples_a = self.load_batch(self.files_testA, batch_size, augment=False)
        samples_b = self.load_batch(self.files_testB, batch_size, augment=False)

        return samples_a, samples_b

    def partition_files(self, batch_size):
        """
        Shuffles the files and partitions them into chunks of length batch_size.
        Returns a list of pairs (a_files, b_files).
        """

        random.shuffle(self.files_trainA)
        random.shuffle(self.files_trainB)
        a_files = chunks(self.files_trainA, batch_size)
        b_files = chunks(self.files_trainB, batch_size)

        return list(zip(a_files, b_files))

    def load_image_batch(self, a_files, b_files):
        size = len(a_files)
        samples_a = self.load_batch(a_files, size)
        samples_b = self.load_batch(b_files, size)

        return samples_a, samples_b


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """

    for i in range(0, len(l), n):
        yield l[i:i + n]
