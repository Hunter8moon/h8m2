import os
from random import randint, random

import numpy as np
from PIL import Image


class ImageUtil:
    @staticmethod
    def file_to_array(file_path, width, height, augment=True):
        """
        Loads a image from disk and returns an np.array pf that image.

        :param file_path: Path to the image to load.
        :param width: Width to resize the image to.
        :param height: Height to resize the image to.
        :param augment: Wether to randomly crop the image or not.
        :return: An np.array of the image.
        """

        im = Image.open(file_path)
        im = im.convert('RGB')

        if augment:
            im = ImageUtil.augment(im, height, width)

        if im.size != (width, height):
            im = im.resize((width, height))

        img = np.asarray(im, dtype=np.uint8)
        img = img / 127.5 - 1.
        return img

    @staticmethod
    def augment(im, height, width):
        x_add = randint(0, 30)
        y_add = randint(0, 30)
        x = randint(0, x_add)
        y = randint(0, y_add)

        # TODO flipping?
        if random() >= 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        im = im.resize((width + x_add, height + y_add))
        im = im.crop((x, y, x + width, y + height))
        return im

    @staticmethod
    def array_to_image(image_array):
        """
        Converts an np.array to a PIL Image.

        :param image_array:  np.array of the image.
        :return: An PIL image of the array.
        """
        img = image_array
        img = img * 127.5 + 127.5
        img = img.astype(np.uint8)
        return Image.fromarray(img)

    @staticmethod
    def save(image, out_dir, filename):
        """
        Saves the image in .png format.
        :param image: The PIL image to save.
        :param out_dir: The directory to save to. Will be created if it does not exist.
        :param filename: The filename of the image.
        """

        directory = f"{out_dir}/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        image.save(f"{directory}{filename}.png")

    @staticmethod
    def make_snapshot_image(images, width, height):
        """
        Lays a 2d array of images out in a grid.

        :param images: The 2d array of images.
        :param width: The width of the images
        :param height: The height of the images.
        :return: An PIL image of the images layed out in a grid.
        """

        n_r = len(images)
        n_c = len(images[0])
        snapshot = Image.new('RGB', (width * n_c, height * n_r))

        r = 0
        for imgs in images:
            c = 0
            for img in imgs:
                img = ImageUtil.array_to_image(img[0])
                snapshot.paste(img, (width * c, height * r))
                c += 1
            r += 1

        return snapshot
