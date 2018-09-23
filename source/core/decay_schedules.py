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

    def __init__(self, start_value=0.0002, min_value=0.00001, cycle_length=25, cycle_multiplier=2, decay_rate=0.75):
        super().__init__()
        self.max = start_value
        self.min = min_value

        self.vals = {}

        self.idx = 0
        self.start = 0
        self.next = cycle_length

        self.cycle_length = cycle_length
        self.cycle_multiplier = cycle_multiplier
        self.decay_rate = decay_rate

        cycle_idx = 0
        start = 0
        next = cycle_length
        cycle_length = cycle_length
        cycle_multiplier = cycle_multiplier
        decay_rate = decay_rate

        for i in range(10_000):
            m = (start_value - min_value) / 2.0

            t_cur = i - start
            t = (t_cur / cycle_length)
            current_value = min_value + m * (1 + math.cos(t * math.pi))

            if i == next:
                cycle_idx += 1
                start = i
                next += (cycle_length * cycle_multiplier)
                cycle_length *= cycle_multiplier

            current_value = current_value * (decay_rate ** cycle_idx)

            self.vals[i] = current_value

    # def f(self, epoch):
    #     epoch = int(epoch)
    #
    #     m = (self.max - self.min) / 2.0
    #
    #     t_cur = epoch - self.start
    #     t = (t_cur / self.cycle_length)
    #     current_value = self.min + m * (1 + math.cos(t * math.pi))
    #
    #     if epoch == self.next:
    #         self.idx += 1
    #         self.start = epoch
    #         self.next += (self.cycle_length * self.cycle_multiplier)
    #         self.cycle_length *= self.cycle_multiplier
    #
    #     current_value = current_value * (self.decay_rate ** self.idx)
    #     return current_value

    def update(self, epoch):
        current_value = self.vals[epoch]

        if self.variables and isinstance(self.variables[0], Variable):
            for var in self.variables:
                backend.set_value(var, current_value)
