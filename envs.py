import gym
import cv2

import numpy as np

from abc import abstractmethod
from collections import deque
from copy import copy

import gym_super_mario_bros
# from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv # line below is the new way to import: https://github.com/uvipen/Super-mario-bros-A3C-pytorch/issues/3
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from PIL import Image
import matplotlib.pyplot as plt

from utils import Logger
import time

train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self, seed):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """
        It returns only every skip-th frame. Action are repeated and rewards are sum for the skipped frames.
        It also takes element-wise maximum over the last two consecutive frames, which helps algorithm deal with the 
        problem of how certain Atari games only render their sprites every other game frame.
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

        # custom rendering:
        if self.is_render:
            self.env.reset()
            plt.ion()
            rendering_view = self.env.render()
            self.ax = plt.imshow(rendering_view)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, _, info = self.env.step(action)
            if self.is_render:
                # custom rendering:
                rendering_view = self.env.render()
                self.ax.set_data(rendering_view)
                plt.xticks([])
                plt.yticks([])
                plt.axis
                plt.pause(1/60)


            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, _, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.is_render:
            # custom rendering:
            rendering_view = self.env.render()
            self.ax.set_data(rendering_view)
            plt.xticks([])
            plt.yticks([])
            plt.axis
            plt.pause(1/60)

        return obs, info


class MaxStepPerEpisodeWrapper(gym.Wrapper):
    def __init__(self, env, max_step_per_episode):
        """
        Let's you call env.step() max_step_per_episode many times before returning done=True, truncated=True
        """
        super(MaxStepPerEpisodeWrapper, self).__init__(env)
        self.max_step_per_episode = max_step_per_episode
        self.steps = 0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.steps += 1
        if self.max_step_per_episode <= self.steps:
            done = True
            truncated = True

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, history_size):
        super().__init__(env)
        assert history_size > 1, "history size must be higher than 1"
        self.history_size = history_size
        self.history = np.zeros((history_size, ) + self.env.observation_space.shape)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        self.history[:(self.history_size - 1)] = self.history[1:, :, :]
        self.history[(self.history_size - 1)] = state

        return self.history, reward, done, truncated, info

    def reset(self, **kwargs):
        state, info =  self.env.reset(**kwargs)
        for i in range(self.history_size):
            self.history[i] = state
        return self.history, info


class StickyActionWrapper(gym.Wrapper):
    def __init__(self, env, p):
        super().__init__(env)
        self.last_action = 0
        self.p = p

    def step(self, action):
        if np.random.rand() <= self.p:
            action = self.last_action
        self.last_action = action
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = 0 
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, _, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, rew, done, _, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class AtariEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=True,
            p=0.25,
            stateStackSize=4,
            seed=42,
            logger:Logger=None):
        super(AtariEnvironment, self).__init__()
        self.daemon = True
        self.env = gym.make(env_id, render_mode="rgb_array" if is_render else None)
        self.env = MaxStepPerEpisodeWrapper(self.env, max_step_per_episode)
        self.env = MaxAndSkipEnv(self.env, is_render)
        if 'Montezuma' in env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_id else 1)
        self.env = Monitor(self.env)
        self.logger = logger
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.seed = seed

        # self.reset(seed=seed)
        self.reset()

    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            if 'Breakout' in self.env_id:
                action += 1

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            s, reward, done, _, info = self.env.step(action)

            self.history[:(self.history_size - 1), :, :] = self.history[1:, :, :]
            self.history[(self.history_size - 1), :, :] = self.pre_proc(s)

            self.rall += reward

            if done:
                self.recent_rlist.append(self.rall)

                if 'Montezuma' in self.env_id:
                    self.logger.log_msg_to_both_console_and_file("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
                        self.episode, self.env_idx, self.env.steps, self.rall, np.mean(self.recent_rlist),
                        info.get('episode', {}).get('visited_rooms', {})))
                else:
                    self.logger.log_msg_to_both_console_and_file("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(
                        self.episode, self.env_idx, self.env.steps, self.rall, np.mean(self.recent_rlist)))

                self.history, _info = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, done, _, info])

    def reset(self, **kwargs):
        self.last_action = 0
        self.episode += 1
        self.rall = 0
        s, info = self.env.reset(**kwargs)
        self.get_init_state(s)
        return self.history[:, :, :], info

    def pre_proc(self, X):
        assert X.shape[-1] == 3 # [H, W, 3]
        x = np.array(Image.fromarray(X).convert('L').resize((self.w, self.h))).astype('float32')
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


class MarioEnvironment(Process):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            life_done=False,
            h=84,
            w=84, movement=COMPLEX_MOVEMENT, sticky_action=True,
            p=0.25,
            seed=42,
            logger:Logger=None):
        super(MarioEnvironment, self).__init__()
        self.daemon = True
        self.logger = logger
        self.env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, render_mode = "rgb_array")
        self.env = MaxStepPerEpisodeWrapper(self.env, max_step_per_episode)
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
        self.env = MaxAndSkipEnv(self.env, is_render)
        self.env = Monitor(self.env)

        self.is_render = is_render
        self.env_idx = env_idx
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.life_done = life_done
        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.seed = seed
        # self.reset(seed=self.seed)
        self.reset()


    def run(self):
        super(MarioEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            obs, r, done, _, info = self.env.step(action)


            # when Mario loses life, changes the state to the terminal
            # state.
            force_done = _
            if self.life_done:
                if self.lives > info['life'] and info['life'] > 0:
                    force_done = True
                    done = True
                    self.lives = info['life']
                else:
                    self.lives = info['life']

            # reward range -15 ~ 15
            r /= 15
            self.rall += r

            # r_ = int(info.get('flag_get', False)) #TODO: not sure how to use this

            self.history[:(self.history_size - 1), :, :] = self.history[1:, :, :]
            self.history[(self.history_size - 1), :, :] = self.pre_proc(obs)


            if done:
                self.recent_rlist.append(self.rall)
                self.logger.log_msg_to_both_console_and_file(
                    "[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Stage: {} current x:{}   max x:{}".format(
                        self.episode,
                        self.env_idx,
                        self.env.steps,
                        self.rall,
                        np.mean(
                            self.recent_rlist),
                        info['stage'],
                        info['x_pos'],
                        self.max_pos))

                self.history, _info = self.reset()

            self.child_conn.send([self.history[:, :, :], r, done, force_done, info])

    def reset(self, **kwargs):
        self.last_action = 0
        self.episode += 1
        self.rall = 0
        self.lives = 3
        self.stage = 1
        self.max_pos = 0
        obs, info = self.env.reset(**kwargs)
        self.get_init_state(obs)

        if self.is_render:
            # custom rendering
            rendering_view = self.env.render()
            self.ax.set_data(rendering_view)
            plt.xticks([])
            plt.yticks([])
            plt.axis
            plt.pause(1/60)
            # self.env.render()

        return self.history[:, :, :], info

    def pre_proc(self, X):
        assert X.shape[-1] == 3 # [H, W, 3]
        x = np.array(Image.fromarray(X).convert('L').resize((self.w, self.h))).astype('float32')
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)

    
class RGBArrayAsObservationWrapper(gym.Wrapper):
    """
    Uses env.render(rgb_array) as observation (image observations)
    rather than the observation environment provides.
    """
    def __init__(self, env):
        super(RGBArrayAsObservationWrapper, self).__init__(env)
        self.env.reset()
        dummy_obs = env.render()
        self.observation_space  = gym.spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.env.render()
        assert obs.shape[-1] == 3
        return obs, info

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        obs = self.env.render()
        assert obs.shape[-1] == 3
        return obs, reward, done, _, info


class ClassicControlEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=True,
            p=0.25,
            stateStackSize=4,
            seed=42,
            logger:Logger=None):
        super(ClassicControlEnvironment, self).__init__()
        self.daemon = True
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env = MaxStepPerEpisodeWrapper(self.env, max_step_per_episode)
        self.env = RGBArrayAsObservationWrapper(self.env)
        self.env = MaxAndSkipEnv(self.env, is_render)
        self.env = Monitor(self.env)

        self.logger = logger
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.seed = seed

        # self.env.seed(seed)
        self.reset()
        # self.reset(seed=seed)

    def run(self):
        super(ClassicControlEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            s, reward, done, _, info = self.env.step(action)

            self.history[:(self.history_size - 1), :, :] = self.history[1:, :, :]
            self.history[(self.history_size - 1), :, :] = self.pre_proc(s)

            self.rall += reward

            if done:
                self.recent_rlist.append(self.rall)
                self.logger.log_msg_to_both_console_and_file("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}".format(
                    self.episode, self.env_idx, self.env.steps, self.rall, np.mean(self.recent_rlist)))

                self.history, _info = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, done, _, info])


    def reset(self, **kwargs):
        self.last_action = 0
        self.episode += 1
        self.rall = 0
        # s = self.env.reset(seed=seed)
        s, info = self.env.reset(**kwargs)
        self.get_init_state(s)
        return self.history[:, :, :], info

    def pre_proc(self, X):
        assert X.shape[-1] == 3 # [H, W, 3]
        x = np.array(Image.fromarray(X).convert('L').resize((self.w, self.h))).astype('float32')
        return x


    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)


class Monitor(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)
        self.tstart = time.time()

        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0

    def reset(self, **kwargs):
        self.rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        ob, rew, done, _, info = self.env.step(action)
        self.rewards.append(rew)
        if done:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"undiscounted_episode_return": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            if "episode" not in info:
                info["episode"] = {}
            info['episode'].update(epinfo)
        self.total_steps += 1
        return (ob, rew, done, _, info)

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times