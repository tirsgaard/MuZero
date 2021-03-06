import gym
import numpy as np
from skimage.transform import resize
from collections import deque, defaultdict


# Function for creating environment. Needs to create seperate env for each worker
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew):
        # modify rew
        return np.float32(rew)


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def observation(self, obs):
        # modify rew
        return obs.reshape((1, 2, 2)).astype(np.float32)


class ReduceImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def observation(self, obs):
        # modify rew
        img = resize(obs, (96, 96))
        img = np.moveaxis(img, 2, 0).astype(np.float32)
        return img


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # Initialise a double ended queue that can store a maximum of two states
        self.obs_buffer = np.zeros((2, 210, 160, 3), dtype=np.float32)
        self.step_number = 0
        self.skip = skip
        self.obs_array = np.empty((self.skip, 90, 90, 3), dtype=np.float32)

    def step(self, action):
        total_reward = 0.0
        for i in range(self.skip):
            # Take a step
            obs, reward, done, info = self.env.step(action)
            # Append the new state to the double ended queue buffer
            self.obs_buffer[self.step_number % 2] = obs
            self.obs_array[i] = resize(np.max(self.obs_buffer, axis=0), (90, 90))
            # Update the total reward by summing the (reward obtained from the step taken) + (the current
            # total reward)
            total_reward += reward
            # If the game ends, break the for loop
            self.step_number += 1
            if done:
                # Fill remaining images with newest observation
                for j in range(i+1, self.skip):
                    self.obs_array[j] = self.obs_array[j - 1]
                break
        max_frames = np.moveaxis(self.obs_array, 3, 1).astype(np.float32)
        max_frames = max_frames.reshape((self.skip*3, 90, 90))
        return max_frames, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = resize(obs.astype(np.float32), (90, 90))
        obs_array = np.repeat(obs[None], self.skip, axis=0)
        obs_array = np.moveaxis(obs_array, 3, 1)
        obs_array = obs_array.reshape((self.skip*3,) + (90, 90))
        return obs_array


class RAMAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4, bits=True):
        super(RAMAndSkipEnv, self).__init__(env)
        # Initialise a double ended queue that can store a maximum of two states
        self.skip = skip
        self.bits = bits
        if bits:
            self.obs_array = np.empty((self.skip, 32, 32), dtype=np.float32)  # 128 bytes -> 1024 bits -> 32*32 values
        else:
            self.obs_array = np.empty((self.skip, 128), dtype=np.float32)  # 128 bytes
        # Check for no-ops in the beginning
        self.no_ops = 0
        self.other_ops = False

    def step(self, action):
        # Check for no ops
        self.other_ops = self.other_ops or (action == 1)  # Store if other action have been performed
        self.no_ops += (action != 1)*(not self.other_ops)  # Store number of other ops
        total_reward = 0.0
        for i in range(self.skip):
            # Take a step
            obs, reward, done, info = self.env.step(action)
            # Append the new state to the double ended queue buffer
            if self.bits:
                obs = np.unpackbits(obs).reshape(32, 32)
            self.obs_array[i] = obs
            # Update the total reward by summing the (reward obtained from the step taken) + (the current
            # total reward)
            total_reward += reward
            # If the game ends, break the for loop
            if done or (self.no_ops>=30):
                done = True  # Case where ops > 30
                # Fill remaining images with newest observation
                for j in range(i+1, self.skip):
                    self.obs_array[j] = self.obs_array[j - 1]
                break
        frames = self.obs_array.astype(np.float32)/256
        return frames, total_reward, done, info

    def reset(self):
        obs = self.env.reset()
        if self.bits:
            obs = np.unpackbits(obs).reshape(32, 32).astype(np.float32)/256  # /256 for normalisation
        obs_array = np.repeat(obs.astype(np.float32)[None], self.skip, axis=0)
        return obs_array


def RAM_Breakout(render):
    return RAMAndSkipEnv(gym.make("ALE/Breakout-v5", full_action_space=False, obs_type="ram", render_mode=render), skip=2, bits=False)


