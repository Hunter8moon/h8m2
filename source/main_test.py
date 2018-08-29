import os
import sys
from glob import glob

import numpy as np
from PIL import Image
from keras import Model

from util.image_util import ImageUtil
from util.save_util import load_model_checkpoint, Input


def memory_hackermann():
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    c = tf.ConfigProto()
    c.gpu_options.allow_growth = True
    c.gpu_options.per_process_gpu_memory_fraction = 0.6
    set_session(tf.Session(config=c))


def parse_args():
    args = sys.argv
    if len(args) != 4:
        print("Usage: ")
        print(" $ python main_test.py <dir_input> <dir_output> <filename_model>")
        exit(-1)

    dir_input = args[1]
    if not os.path.exists(dir_input):
        print(fr"[Error] Input directory does not exist. ({dir_input}) ")
        exit(-1)
    if not glob(f'{dir_input}/*'):
        print(fr"[Error] Input directory is empty. ({dir_input}) ")
        exit(-1)

    dir_output = args[2]
    if not os.path.exists(dir_output):
        print(fr"[Warning] Output directory does not exist.  ")
        print(fr"          Creating directory ({dir_output}) . . . ")
        os.makedirs(dir_output)

    filename_model = args[3]
    if not os.path.isfile(filename_model):
        print(fr"[Error] Model file does not exist. ({filename_model}) ")
        exit(-1)

    return dir_input, dir_output, filename_model


def test_directory(input_dir, output_dir, model, image_shape):
    """
    :param input_dir:  The input directory of images to be converted.
    :param output_dir:  The output directory where the converted images will be saved.
    """

    idx = 0
    files = glob(f'{input_dir}/*')
    for file in files:
        print(f"[{idx}/{len(files)}]", end='')

        in_img = ImageUtil.file_to_array(file, image_shape[0], image_shape[1], augment=False)
        in_img = np.array([in_img])

        output = model.predict(in_img)
        out_img = ImageUtil.array_to_image(output[0])

        filename = os.path.splitext(os.path.basename(file))[0]
        ImageUtil.save(out_img, output_dir, filename)

        idx += 1
        print(f" Done saving: {filename}")


def make_input_size_flexible(model):
    model.layers.pop(0)

    input_layer = Input(batch_shape=(None, None, None, 3))
    output_layer = model(input_layer)
    out_model = Model(input_layer, output_layer)

    return out_model


def get_image_shape(dir_input):
    file = glob(f'{dir_input}/*')
    img = Image.open(file[0])
    width, height = img.size
    return width, height, 3


if __name__ == '__main__':
    # In case of CUDNN_STATUS_INTERNAL_ERROR, try this:
    # memory_hackermann()

    dir_input, dir_output, filename_model = parse_args()

    gen = load_model_checkpoint(filename_model)
    gen = make_input_size_flexible(gen)
    image_shape = get_image_shape(dir_input)

    test_directory(dir_input, dir_output, gen, image_shape)
