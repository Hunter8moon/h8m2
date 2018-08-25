import tensorflow as tf
from keras.layers import ZeroPadding2D


class ReflectionPadding2D(ZeroPadding2D):
    """
    From:
    https://github.com/robertomest/neural-style-keras/blob/c3a9836c9a18a7bf74b1de4588b6a5a138b3162a/layers.py
    """

    def __init__(self, padding=((1, 1), (1, 1)), **kwargs):
        super().__init__(padding, **kwargs)

        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        if isinstance(padding[0], int):
            self.padding = ((padding[0], padding[0]), (padding[1], padding[1]))

        self.top_pad = padding[0][0]
        self.bottom_pad = padding[0][1]
        self.right_pad = padding[1][0]
        self.left_pad = padding[1][1]

    def call(self, x, **kwargs):
        pattern = [[0, 0],
                   [self.top_pad, self.bottom_pad],
                   [self.left_pad, self.right_pad],
                   [0, 0]]
        return tf.pad(x, pattern, mode='REFLECT')
