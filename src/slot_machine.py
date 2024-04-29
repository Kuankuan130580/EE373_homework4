import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
# import src.random

class SlotMachine:
    def __init__(self, mean, std_dev):
        self.mean = mean
        self.std_dev = std_dev

    def pull(self):
        return np.random.normal(loc=self.mean, scale=self.std_dev)

class SlotMachines(gymnasium.Env):
    """
    Slot machine reinforcement learning environment for OpenAI Gym

    Arguments:
        n_machines - (int) Number of slot machines to create
        mean_range - (tuple) Range of values for mean initialization
        std_range - (tuple) Range of values for std initialization
    """

    def __init__(self, mean_list= [], std_dev=[]):
        self.machines = [SlotMachine(mean_list[i],std_dev[i]) for i in range(len(mean_list))]        # Required by OpenAI Gym
        n_machines = len(mean_list)
        self.action_space = spaces.Discrete(n_machines)
        self.observation_space = spaces.Discrete(1)

    def seed(self, seed=None):
        """
        Seed the environment's random number generator
        Arguments:
          seed - (int) The random number generator seed.
        """
        _, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Perform an action within the slot machine environment

        Arguments:
          action - (int) An action to perform

        Returns:
          observation - (int) The new environment state. This is always 0 for
            SlotMachines.
          reward - (float) The reward gained by taking an action.
          terminated - (bool) Whether the environment has been completed and requires
            resetting. This is always True for SlotMachines.
          truncated - (bool) Whether the environment has been completed and requires
            resetting. This is always True for SlotMachines.
          info - (dict) A dictionary of additional return values used for
            debugging purposes.
        """
        assert self.action_space.contains(action)
        return 0, self.machines[action].pull(), True, True, {}

    def reset(self):
        """
        Resets the environment. For SlotMachines, this always returns 0.
        """
        return 0, {'prob': 1}

    def render(self, mode='human', close=False):
        """
        Render the environment display. For SlotMachines, this is a no-op.
        """
        pass