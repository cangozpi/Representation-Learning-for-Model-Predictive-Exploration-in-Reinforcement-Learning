import gym
from envs import AtariEnvironment
from multiprocessing import Pipe
from envs import MaxStepPerEpisodeWrapper, MaxAndSkipEnv, MontezumaInfoWrapper, FrameStackWrapper, StickyActionWrapper, Monitor
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import play

def test_Atari():
    default_config = {'EnvType': 'atari'}
    env_id = "MontezumaRevengeNoFrameskip-v4"

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    elif default_config['EnvType'] == 'classic_control':
        env_type = ClassicControlEnvironment
    else:
        raise NotImplementedError

    is_render = False
    sticky_action = False
    action_prob = 0.5
    life_done = True
    stateStackSize = 4
    seed = 42
    from utils import Logger
    from os import path
    logger = Logger(file_log_path=path.join("logs", "file_logs", "test_log"), tb_log_path=path.join("logs", "tb_logs", "test_log"))
    num_worker = 1

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, history_size=stateStackSize, seed=seed, logger=logger)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
    
    # Env is ready here


def get_env(env_id = "MontezumaRevengeNoFrameskip-v4", **kwargs):
    env = gym.make(env_id, **kwargs)
    return env


def test_MaxStepPerEpisodeWrapper():
    max_step_per_episode_list = [4, 8, 12]

    def test_max_step_per_episode(max_step_per_episode):
        env = get_env()
        env = MaxStepPerEpisodeWrapper(env, max_step_per_episode)
        state, info = env.reset()
        for i in range(max_step_per_episode):
            a = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(a)

            if done and i != (max_step_per_episode - 1): # episode terminated before reaching max steps
                assert truncated == False
                state, info = env.reset()
                test_max_step_per_episode(max_step_per_episode)
                break

            # print(f'env.steps: {env.steps}, env.max_step_per_episode: {env.max_step_per_episode}, self.max_step_per_episode <= self.steps: {env.max_step_per_episode <= env.steps}')
            if i != max_step_per_episode - 1:
                assert truncated == False
            else:
                assert done == True
                assert truncated == True

    for max_step_per_episode in max_step_per_episode_list:
        print(f'testing with max_step_per_episode: {max_step_per_episode}')
        test_max_step_per_episode(max_step_per_episode)


def test_MaxAndSkipEnvWrapper(render=False):
    """
    render (bool): set to True for visually inspect the recent frames
    """
    skip = 4
    class MonitorRecentSkipFramesWrapper(gym.Wrapper):
        """
        Keeps a record of the latest skip frames in self.recent_frames.
        """
        def __init__(self, env, skip):
            super().__init__(env)
            self.skip = skip
            self.recent_frames = np.zeros((skip, ) + env.observation_space.shape, dtype=np.uint8)

        def step(self, action):
            obs, reward, done, truncated, info = self.env.step(action)
            self.recent_frames[:(self.skip-1), :, :] = self.recent_frames[1:, :, :]
            self.recent_frames[self.skip-1, :, :] = obs
            return obs, reward, done, truncated, info
        
        def reset(self, **kwargs):
            return self.env.reset(**kwargs)

    env = get_env()
    env = MonitorRecentSkipFramesWrapper(env, skip)
    env = MaxAndSkipEnv(env, is_render=False, skip=skip)
    state, info = env.reset()

    for i in range(100):
        a = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(a)

        if render:
            fig, axs = plt.subplots(skip, 2, constrained_layout=True)
            fig.set_figheight(8)
            fig.set_figwidth(8)
            for j in range(skip):
                axs[j, 0].imshow(env.recent_frames[j]) # [H, W, C=3]
                axs[j, 0].set_title(f'recent frame: {j} (step: {i})')
                axs[j, 0].tick_params(top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False)

            for j in range(skip):
                axs[j, 1].imshow(next_state) # [H, W, C=3]
                axs[j, 1].set_title(f'{skip}-th Max frame (step: {i})')
                axs[j, 1].tick_params(top=False,
                bottom=False,
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False)
            plt.show()

        assert np.allclose(env.recent_frames[-2:], env._obs_buffer), f'recent 2 frames at step:{i} does not match !'
        assert np.allclose(np.max(env.recent_frames[-2:], axis=0), next_state), f'Issue with Maxing Frames at step:{i}'

        if done or (i % 50 == 0): # also reset at some point to check that env.reset does not break anything
            state, info = env.reset()
    

def test_MontezumaInfoWrapper():
    class VisitedRoomPrinterWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.latest_visited_rooms = set()
        def step(self, action):
            state, reward, done, truncated, info = self.env.step(action)
            if len(self.env.visited_rooms.difference(self.latest_visited_rooms)) > 0:
                print(self.latest_visited_rooms.difference(self.env.visited_rooms))
                print(f'visited_room: {self.env.visited_rooms}')
                from copy import copy
                self.latest_visited_rooms = copy(self.env.visited_rooms)
            return state, reward, done, truncated, info
        def reset(self, **kwargs):
            state, info = self.env.reset(**kwargs)
            if len(self.env.visited_rooms.difference(self.latest_visited_rooms)) > 0:
                print(f'visited_room: {self.env.visited_rooms}')
                self.latest_visited_rooms = copy(self.env.visited_rooms)
            return state, info

    env = get_env(env_id = "MontezumaRevengeNoFrameskip-v4", render_mode="rgb_array")
    env = MontezumaInfoWrapper(env, room_address=3)
    env = VisitedRoomPrinterWrapper(env)
    print("play and explore rooms by clicking on w, a, s, d, space_bar\n when you discover new rooms it will be printed here ...")
    play.play(env, zoom=3)


def test_FrameStackWrapper():
    history_size = 4
    env = get_env()
    env = FrameStackWrapper(env, history_size)
    state, info = env.reset()
    fig, axs = plt.subplots(history_size, constrained_layout=True, figsize=(6, 8))
    plt.ion()
    for i in range(100):
        for j in range(history_size):
            axs[j].imshow(state[j].astype(np.uint8))
            axs[j].set_title(f'stacked frame: t-{history_size-j-1}')
            axs[j].tick_params(top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False)
        plt.pause(1/60)
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        if done:
            state, info = env.reset()


def test_StickyActionWrapper():
    # Env should always take the env.latest_action = 0
    p = 1
    env = get_env()
    env = StickyActionWrapper(env, p)
    state, info = env.reset()
    for i in range(10):
        action = 0
        while action == env.last_action:
            action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        # print(f'sent action: {action}, env took stickyAction: {env.last_action}')
        assert env.last_action == 0

    # Env should always take the sent action:
    p = 0
    env = get_env()
    env = StickyActionWrapper(env, p)
    state, info = env.reset()
    for i in range(10):
        action = 0
        while action == env.last_action:
            action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        # print(f'sent action: {action}, env took stickyAction: {env.last_action}')
        assert env.last_action == action

def test_MonitorWrapper():
    env = get_env()
    env = Monitor(env)
    done = False
    state, info = env.reset()
    undiscounted_episode_return = 0
    episode_length = 0
    while done == False:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        undiscounted_episode_return += reward
        episode_length += 1
    assert undiscounted_episode_return == info['episode']['undiscounted_episode_return']
    assert episode_length == info['episode']['l']



if __name__ == "__main__":
    # test_MaxStepPerEpisodeWrapper()
    # test_MaxAndSkipEnvWrapper(render=False)
    # test_MontezumaInfoWrapper()
    # test_FrameStackWrapper()
    # test_StickyActionWrapper()
    test_MonitorWrapper()

    # test_Atari()