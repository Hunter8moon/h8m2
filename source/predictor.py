import os
from glob import glob

import numpy as np

from util.image_util import ImageUtil


class Predictor:
    def __init__(self, predictor, image_shape):
        self.image_shape = image_shape
        self.predictor = predictor

    def predict_directory(self, input_dir, output_dir):
        """

        :param input_dir:  The input directory of images to be converted.
        :param output_dir:  The output directory where the converted images will be saved.
        """

        idx = 0
        files = glob(f'{input_dir}/*')
        for file in files:
            print(f"[{idx}/{len(files)}]", end='')

            in_img = ImageUtil.file_to_array(file, self.image_shape[0], self.image_shape[1], augment=False)
            in_img = np.array([in_img])

            output = self.predictor(in_img)
            out_img = ImageUtil.array_to_image(output[0])

            filename = os.path.splitext(os.path.basename(file))[0]
            ImageUtil.save(out_img, output_dir, filename)

            idx += 1
            print(f" Done saving: {filename}")
