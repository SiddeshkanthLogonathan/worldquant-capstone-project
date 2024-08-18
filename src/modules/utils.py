from __future__ import annotations

from collections import deque
from random import randint
from random import random

import numpy as np
from torch.utils.data.dataset import IterableDataset


class PVM:
    def __init__(self, capacity, portfolio_size, multi_period_horizon=1):
        """Initializes portfolio vector memory.

        Args:
          capacity: Max capacity of memory.
          portfolio_size: Portfolio size.
        """
        # initially, memory will have the same actions
        self.capacity = capacity
        self.portfolio_size = portfolio_size
        self.multi_period_horizon = multi_period_horizon
        self.reset()

    def reset(self):
        self.memory = [np.tile([np.array([1] + [0] * self.portfolio_size, dtype=np.float32)], reps=(self.multi_period_horizon, 1))] * (
            self.capacity + 1
        )
        self.index = 0  # initial index to retrieve data

    def retrieve(self):
        last_action = self.memory[self.index]
        self.index = 0 if self.index == self.capacity else self.index + 1
        return last_action

    def add(self, action):
        self.memory[self.index] = action


class ReplayBuffer:
    def __init__(self, capacity):
        """Initializes replay buffer.

        Args:
          capacity: Max capacity of buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        """Represents the size of the buffer

        Returns:
          Size of the buffer.
        """
        return len(self.buffer)

    def append(self, experience):
        """Append experience to buffer. When buffer is full, it pops
           an old experience.

        Args:
          experience: experience to be saved.
        """
        self.buffer.append(experience)

    def sample(self):
        """Sample from replay buffer. All data from replay buffer is
        returned and the buffer is cleared.

        Returns:
          Sample of batch_size size.
        """
        buffer = list(self.buffer)
        self.buffer.clear()
        return buffer


class RLDataset(IterableDataset):
    def __init__(self, buffer):
        """Initializes reinforcement learning dataset.

        Args:
            buffer: replay buffer to become iterable dataset.

        Note:
            It's a subclass of pytorch's IterableDataset,
            check https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        self.buffer = buffer

    def __iter__(self):
        """Iterates over RLDataset.

        Returns:
          Every experience of a sample from replay buffer.
        """
        yield from self.buffer.sample()


def apply_portfolio_noise(portfolio, epsilon=0.0):
    """Apply noise to portfolio distribution considering its constrains.

    Arg:
        portfolio: initial portfolio distribution.
        epsilon: maximum rebalancing.

    Returns:
        New portolio distribution with noise applied.
    """
    portfolio_shape = portfolio.shape
    portfolio_size = portfolio_shape[1]
    new_portfolio = portfolio.copy()
    for t in range(portfolio_shape[0]): # Time Component
        for i in range(portfolio_shape[1]): # Stocks Component
            target_index = randint(0, portfolio_size - 1)
            difference = epsilon * random()
            # check constrains
            max_diff = min(new_portfolio[t,i], 1 - new_portfolio[t,target_index])
            difference = min(difference, max_diff)
            # apply difference
            new_portfolio[t,i] -= difference
            new_portfolio[t,target_index] += difference
    return new_portfolio

def calculate_max_draw_down(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    return drawdowns