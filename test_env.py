import gym
from gym import error, spaces, utils
import numpy as np
from gym.utils import seeding

class testEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    def __init__(self):
        """
        Every environment should be derived from gym.Env and at least contain the variables observation_space and action_space
        specifying the type of possible observations and actions using spaces.Box or spaces.Discrete.
        Example:
        >>> EnvTest = FooEnv()
        >>> EnvTest.observation_space=spaces.Box(low=-1, high=1, shape=(3,4))
        >>> EnvTest.action_space=spaces.Discrete(2)
        """
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2, 2))
        self.max_lives = 1
        self.step_number = 0
        self.lives = self.max_lives
        self.max_length = 5
        self.done = False

    def return_obs(self):
        S = np.zeros((2, 2), dtype=np.float32)
        S[0, 0] = self.step_number
        S[0, 1] = self.lives
        S[1, :] = np.random.rand(2)
        return S

    def step(self, action):
        """
        This method is the primary interface between environment and agent.
        Paramters:
            action: int
                    the index of the respective action (if action space is discrete)
        Returns:
            output: (array, float, bool)
                    information provided by the environment about its current state:
                    (observation, reward, done)
        """
        if not self.done:
            correct_choice = self.step_number % 2 == action
            reward = float(correct_choice)
            self.lives -= float(not correct_choice)  # If incorrect choice is made, loose 1 life
            self.step_number += 1

            observation = self.return_obs()
            self.done = (self.step_number == self.max_length) or (self.lives == 0)
            return observation, reward, self.done, None
        else:
            return


    def reset(self):
        """
        This method resets the environment to its initial values.
        Returns:
            observation:    array
                            the initial state of the environment
        """
        self.step_number = 0
        self.lives = self.max_lives
        self.done = False
        S = self.return_obs()
        return S

    def render(self, mode='human', close=False):
        """
        This methods provides the option to render the environment's behavior to a window
        which should be readable to the human eye if mode is set to 'human'.
        """
        pass

    def close(self):
        """
        This method provides the user with the option to perform any necessary cleanup.
        """
        pass