import random
from collections import deque

import numpy as np


class ImageBuffer:
    def __init__(self, buffer_size, batch_size):
        self.size = buffer_size
        self.buffer_size = batch_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, images):
        random.shuffle(self.buffer)
        self.buffer.extend(images)

    def get_batch(self):
        batch_size = min(self.buffer_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        return np.array(batch)
