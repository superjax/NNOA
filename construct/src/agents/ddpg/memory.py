"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""
import random
import numpy as np
from threading import Lock
from collections import deque
from construct import GazeboEnvironment

class Memory:
    def __init__(self, phi_frames, replay_memory_size, environment):
        odo_space, screen_space = environment.observation_space.spaces

        self.actions = np.empty(tuple([replay_memory_size]) + environment.action_space.sample().shape, dtype=np.float32)
        self.rewards = np.empty(replay_memory_size, dtype=np.float64)
        self.priorities = np.zeros(replay_memory_size, dtype=np.float64)

        self.screen_dims = screen_space.shape
        self.odometry_dims = odo_space.shape
        self.screens = np.empty(tuple([replay_memory_size]) + self.screen_dims, dtype=np.uint8)
        self.odometry = np.empty(tuple([replay_memory_size]) + self.odometry_dims, dtype=np.uint8)
        self.terminals = np.empty(replay_memory_size, dtype=np.bool)

        self.count = 0
        self.current = 0
        self.priority_sum = 0
        self.priority_exp_avg = 0

        self.void_phi = np.zeros(tuple([phi_frames]) + screen_space.sample().shape, dtype=np.uint8)

        self.replay_memory_size = replay_memory_size
        self.phi_frames = phi_frames
        self.use_prioritization = True

    def update(self, index, priority):
        if self.use_prioritization:
            for si, i in enumerate(index):
                self.priority_sum -= self.priorities[i]
                self.priorities[i] = priority[si]
                self.priority_sum += priority[si]
                self.priority_exp_avg -= (1 - .99999) * (self.priority_exp_avg - priority[si])

    def add(self, state, reward, action, terminal, priority=None):
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.odometry[self.current, ...] = state[0]
        self.screens[self.current, ...] = state[1]
        self.terminals[self.current] = terminal

        # TODO: this is bad, we need to figure out what to do with priority-less entries. maybe every batch has 1 random element with 0 probability?
        self.update([self.current], [priority if priority is not None else 300])

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.replay_memory_size

    def get_state(self, index):
        assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.phi_frames - 1:
            # use faster slicing
            return self.odometry[(index - (self.phi_frames - 1)):(index + 1), ...], self.screens[(index - (self.phi_frames - 1)):(index + 1), ...],
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.phi_frames))]
            return self.odometry[indexes, ...], self.screens[indexes, ...]

    def can_sample(self, batch_size):
        return self.count > batch_size

    def sample_priority_indexes(self, size):
        cumsum = np.cumsum(self.priorities[0:self.count])
        proposal = np.random.random(size) * cumsum[-1]
        # Faster than np.random.choice
        return list(np.searchsorted(cumsum, proposal, side='left'))

    def sample(self, batch_size):
        # memory must include poststate, prestate and history
        assert self.can_sample(batch_size)

        indexes = []
        random_indexes = []

        screen_prestates = np.zeros((batch_size, self.phi_frames) + self.screen_dims, dtype=self.screens.dtype)
        screen_poststates = np.zeros((batch_size, self.phi_frames) + self.screen_dims, dtype=self.screens.dtype)

        odometry_prestates = np.zeros((batch_size, self.phi_frames) + self.odometry_dims, dtype=self.odometry.dtype)
        odometry_poststates = np.zeros((batch_size, self.phi_frames) + self.odometry_dims, dtype=self.odometry.dtype)

        if self.use_prioritization:
            random_indexes = self.sample_priority_indexes(batch_size)

        while len(indexes) < batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                if len(random_indexes) > 0:
                    index = random_indexes.pop()
                else:
                    index = random.randint(self.phi_frames, self.count - 1)

                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.phi_frames < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.phi_frames):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            odometry_prestates[len(indexes), ...], screen_prestates[len(indexes), ...] = self.get_state(index - 1)
            odometry_poststates[len(indexes), ...], screen_poststates[len(indexes), ...] = self.get_state(index)

            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return [odometry_prestates, screen_prestates], actions, rewards, [odometry_poststates, screen_poststates], terminals, indexes