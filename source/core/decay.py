import math

from keras import backend
from tensorflow import Variable


class Schedule:
    def __init__(self):
        self.variables = []

    def update(self, epoch):
        raise NotImplementedError

    def bind_to_variables(self, variables):
        self.variables.extend(variables)


class IdentitySchedule(Schedule):
    def __init__(self, value=1):
        super().__init__()
        self.value = value

    def update(self, epoch):
        if self.variables and isinstance(self.variables[0], Variable):
            for var in self.variables:
                backend.set_value(var, self.value)


class LinearSchedule(Schedule):
    """
    Linear decay.
    """

    def __init__(self, start_value, end_value=0, start_epoch=100, end_epoch=200):
        super().__init__()

        self.start_value = start_value
        self.end_value = end_value
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

        self.total_epochs = self.end_epoch - self.start_epoch
        self.step = (self.end_value - self.start_value) / self.total_epochs

    def update(self, epoch):
        if epoch < self.start_epoch:
            current_value = self.start_value
        elif epoch >= self.end_epoch:
            current_value = self.end_value
        else:
            epoch = epoch - self.start_epoch
            current_value = self.start_value + (self.step * epoch)

        if self.variables and isinstance(self.variables[0], Variable):
            for var in self.variables:
                backend.set_value(var, current_value)


class WarmRestart(Schedule):
    """
    Cyclic learning rate.
    From: https://arxiv.org/abs/1608.03983
    """

    def __init__(self, max_value=0.0002, min_value=0.00001, cycle_length=10, cycle_multiplier=2, decay_rate=0.9):
        super().__init__()
        self.max = max_value
        self.min = min_value
        self.t = cycle_length
        self.t_mult = cycle_multiplier
        self.decay_rate = decay_rate

    def update(self, epoch):
        t_cur = epoch % self.t
        current_value = self.min + 0.5 * (self.max - self.min) * (1 + math.cos((t_cur / self.t) * math.pi))

        decay = self.decay_rate ** (epoch // self.t)
        current_value *= decay

        if self.variables and isinstance(self.variables[0], Variable):
            for var in self.variables:
                backend.set_value(var, current_value)
