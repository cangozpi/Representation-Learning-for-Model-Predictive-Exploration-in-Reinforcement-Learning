import gym
from envs import AtariEnvironment, MarioEnvironment, ClassicControlEnvironment
from multiprocessing import Pipe
from envs import MaxStepPerEpisodeWrapper, MaxAndSkipEnv, MontezumaInfoWrapper, FrameStackWrapper, StickyActionWrapper, Monitor, ResizeAndGrayScaleWrapper, RGBArrayAsObservationWrapper
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import play
from utils import ParallelizedEnvironmentRenderer


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


def test_ResizeAndGrayScaleWrapper():
    h, w = 84, 84
    env = get_env()
    env = ResizeAndGrayScaleWrapper(env, h, w)
    state, info = env.reset()
    assert state.shape == (h, w)
    done = False
    while done == False:
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        assert next_state.shape == (h, w)
        state = next_state


def test_RGBArrayAsObservationWrapper():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env = RGBArrayAsObservationWrapper(env)

    state, info = env.reset()
    assert len(state.shape) == 3
    assert state.shape[-1] == 3
    action = env.action_space.sample()
    next_state, reward, done, _, info = env.step(action)
    assert len(next_state.shape) == 3
    assert next_state.shape[-1] == 3


def test_CustomEnvironments(env_type='atari', env_id='MontezumaRevengeNoFrameskip-v4'):
    default_config = {'EnvType': env_type}
    env_id = env_id

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    elif default_config['EnvType'] == 'classic_control':
        env_type = ClassicControlEnvironment
    else:
        raise NotImplementedError

    every_env_takes_same_action = True # if True then every env takes the same action, if False then actions are sampled independently for each parallelized env
    is_render = False
    sticky_action = False
    action_prob = 0.5
    life_done = True
    stateStackSize = 4
    seed = 42
    from utils import Logger
    from os import path
    logger = Logger(file_log_path=path.join("logs", "file_logs", "test_log"), tb_log_path=path.join("logs", "tb_logs", "test_log"))
    num_worker = 2

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, history_size=stateStackSize, seed=seed+idx, logger=logger)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
    
    # Env is ready here
    output_size = works[0].env.action_space.n
    input_size = works[0].env.observation_space.shape[0]
    # fig, axs = plt.subplots(num_worker, figsize=(6, 8), constrained_layout=True)
    # plt.ion()
    renderer = ParallelizedEnvironmentRenderer(num_worker)
    for step in range(1000):
        if every_env_takes_same_action:
            action = np.random.randint(0, output_size, size=(1,))
            actions = np.repeat(action, num_worker)
        else:
            actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_obs = []
        for parent_conn in parent_conns:
            s, r, d, _, info = parent_conn.recv()
            next_obs.append(s[stateStackSize - 1, :, :].reshape([1, input_size, input_size]))

        # for i in range(num_worker):
        #     axs[i].imshow(next_obs[i].squeeze(0).astype(np.uint8), cmap="gray")
        #     axs[i].set_title(f'worker: {i}')
        #     axs[i].tick_params(top=False,
        #     bottom=False,
        #     left=False,
        #     right=False,
        #     labelleft=False,
        #     labelbottom=False)
        # plt.pause(1/60)
        next_obs = np.stack(next_obs) # -> [num_env, 1, H, W]
        renderer.render(next_obs)
        # next_obs = []
    renderer.close()


def test_every_paralell_env_is_playing_different_games(visualize=False, take_same_action=False):
    """
    Check that every parallel env is playing different games. In other words different games btw different parallel envs.
    """
    # Set seed:
    from utils import set_seed
    seed = 42
    set_seed(seed) # Note: this will not seed the gym environment

    # Create parallelized envs:
    from dist_utils import create_parallel_env_processes
    from envs import AtariEnvironment
    num_env_per_process = 2
    env_type = AtariEnvironment
    env_id = 'PongNoFrameskip-v4'
    sticky_action = True
    action_prob = 0.25
    life_done = False
    stateStackSize = 4
    input_size = 84
    output_size = 6
    class DummyLogger:
        def __init__(self):
            pass
        def log_msg_to_both_console_and_file(self, *args, **kwargs):
            # print(*args, **kwargs)
            pass
    logger = DummyLogger()
    _, env_worker_parent_conns, _ = create_parallel_env_processes(num_env_per_process, env_type, env_id, sticky_action, action_prob, life_done, stateStackSize, input_size, seed, logger)

    num_iter = 1000
    next_states = np.zeros([num_iter, num_env_per_process, stateStackSize, input_size, input_size], dtype=np.float64) # -> [num_env, state_stack_size, H, W]
    for i in range(num_iter): # take 10 step rollouts from 2 parallel envs
        # Send actions:
        if not take_same_action:
            actions = np.random.randint(0, output_size, size=(num_env_per_process,)).astype(dtype=np.int64) # Note that random action taking might be an issue which results in same images being used for training
        else:
            actions = np.zeros((num_env_per_process,)).astype(dtype=np.int64)
            actions[:] = np.random.randint(0, output_size, size=(1,)).astype(dtype=np.int64) # this will broadcast to every dimension
        for parent_conn, action in zip(env_worker_parent_conns, actions):
            parent_conn.send(action)

        # Get observations:
        for env_idx, parent_conn in enumerate(env_worker_parent_conns):
            s, r, d, trun, visited_rooms = parent_conn.recv()
            next_states[i, env_idx] = s[:]

            if d or trun:
                info = {'episode': {}}
                if 'Montezuma' in env_id:
                    info['episode']['number_of_visited_rooms'], info['episode']['visited_rooms'] = parent_conn.recv()
                    number_of_visited_rooms.append(info['episode']['number_of_visited_rooms'])
                info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes'] = parent_conn.recv()

        
    # Check if envs play the same game:
    same_frame_count = 0
    import matplotlib.pyplot as plt
    if visualize:
        fig, axs = plt.subplots(2, 4, constrained_layout=True)
        plt.ion()
    for t in range(len(next_states)):
        if visualize:
            axs[0,0].imshow(next_states[t, 0,0,:,:], cmap='gray')
            axs[0,1].imshow(next_states[t, 0,1,:,:], cmap='gray')
            axs[0,2].imshow(next_states[t, 0,2,:,:], cmap='gray')
            axs[0,3].imshow(next_states[t, 0,3,:,:], cmap='gray')

            axs[1,0].imshow(next_states[t, 1,0,:,:], cmap='gray')
            axs[1,1].imshow(next_states[t, 1,1,:,:], cmap='gray')
            axs[1,2].imshow(next_states[t, 1,2,:,:], cmap='gray')
            axs[1,3].imshow(next_states[t, 1,3,:,:], cmap='gray')

            axs[0,0].set_title(f'env:{0}, step:{t}, frame:0', fontsize=10)
            axs[0,1].set_title(f'env:{0}, step:{t}, frame:1', fontsize=10)
            axs[0,2].set_title(f'env:{0}, step:{t}, frame:2', fontsize=10)
            axs[0,3].set_title(f'env:{0}, step:{t}, frame:3', fontsize=10)

            axs[1,0].set_title(f'env:{1}, step: {t}, frame:0', fontsize=10)
            axs[1,1].set_title(f'env:{1}, step: {t}, frame:1', fontsize=10)
            axs[1,2].set_title(f'env:{1}, step: {t}, frame:2', fontsize=10)
            axs[1,3].set_title(f'env:{1}, step: {t}, frame:3', fontsize=10)

            plt.pause(1/60)

        if np.allclose(next_states[t, 0, -1], next_states[t, 1, -1]):
            print(f'frames at step:{t} are the same!')
            same_frame_count += 1

    if same_frame_count > num_iter//10: # take 10 percent of all frames as the threshold
        print(f'[Test] "test_parallelized_env_seeds": FAILED with same_frame_count: {same_frame_count}')
        return False
    else:
        print(f'[Test] "test_parallelized_env_seeds": PASSED, with same_frame_count: {same_frame_count}')
        return True


def test_after_reset_the_same_env_plays_different_games(visualize=False):
    """
    Check that after reset the env is not playing the same game over and over again. In other words different games btw env resets.
    """
    # Set seed:
    from utils import set_seed
    seed = 42
    set_seed(seed) # Note: this will not seed the gym environment

    # Create parallelized envs:
    from dist_utils import create_parallel_env_processes
    from envs import AtariEnvironment
    num_env_per_process = 1
    env_type = AtariEnvironment
    env_id = 'PongNoFrameskip-v4'
    sticky_action = True
    action_prob = 0.25
    life_done = False
    stateStackSize = 4
    input_size = 84
    output_size = 6
    class DummyLogger:
        def __init__(self):
            pass
        def log_msg_to_both_console_and_file(self, *args, **kwargs):
            # print(*args, **kwargs)
            pass
    logger = DummyLogger()
    _, env_worker_parent_conns, _ = create_parallel_env_processes(num_env_per_process, env_type, env_id, sticky_action, action_prob, life_done, stateStackSize, input_size, seed, logger)


    first_reset_states = []
    first_reset_actions = []
    second_reset_states = []
    second_reset_actions = []
    first_reset_flag = False
    second_reset_flag = False
    data_collection_done = False
    action_idx = 0
    while True: # keep collecting rollouts for 2 full episodes
        # Send actions:
        if second_reset_flag == False:
            actions = np.random.randint(0, output_size, size=(num_env_per_process,)).astype(dtype=np.int64) # Note that random action taking might be an issue which results in same images being used for training
            for parent_conn, action in zip(env_worker_parent_conns, actions):
                parent_conn.send(action)
        else:
            if action_idx >= len(first_reset_actions):
                print(f'no more actions left in the first_reset_actions so taking a random action after step: {action_idx}')
                actions = np.random.randint(0, output_size, size=(num_env_per_process,)).astype(dtype=np.int64) # Note that random action taking might be an issue which results in same images being used for training
                for parent_conn, action in zip(env_worker_parent_conns, actions):
                    parent_conn.send(action)
            else:
                for parent_conn, action in zip(env_worker_parent_conns, first_reset_actions[action_idx]):
                    parent_conn.send(action)



        # Get observations:
        next_states = np.zeros([num_env_per_process, stateStackSize, input_size, input_size], dtype=np.float64) # -> [num_env, state_stack_size, H, W]
        for env_idx, parent_conn in enumerate(env_worker_parent_conns):
            s, r, d, trun, visited_rooms = parent_conn.recv()
            next_states[env_idx] = s[:]

            if d or trun:
                print('env.reset() called')
                info = {'episode': {}}
                if 'Montezuma' in env_id:
                    info['episode']['number_of_visited_rooms'], info['episode']['visited_rooms'] = parent_conn.recv()
                    number_of_visited_rooms.append(info['episode']['number_of_visited_rooms'])
                info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes'] = parent_conn.recv()

                if first_reset_flag == False and second_reset_flag == False:
                    first_reset_flag = True
                elif first_reset_flag == True and second_reset_flag == False:
                    second_reset_flag = True
                elif first_reset_flag == True and second_reset_flag == True:
                    data_collection_done = True

            if first_reset_flag == True and second_reset_flag == False: # before first reset
                first_reset_states.append(s)
                first_reset_actions.append(actions)
            elif first_reset_flag == True and second_reset_flag == True and (not data_collection_done): # after first reset
                second_reset_states.append(s)
                if action_idx >= len(first_reset_actions):
                    second_reset_actions.append(actions)
                else:
                    second_reset_actions.append(first_reset_actions[action_idx])
                    action_idx += 1
            
        # check if two different reset rollouts have been collected
        if first_reset_flag and second_reset_flag and data_collection_done:
            break
    
    # Play the two recorded episodes side by side for visual debugging:
    if visualize:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        plt.ion()
        for t in range(len(first_reset_states)):
            axs[0].imshow(first_reset_states[t][-1:,:].transpose(1,2,0), cmap='gray')
            axs[1].imshow(second_reset_states[t][-1:,:].transpose(1,2,0), cmap='gray')


            axs[0].set_title(f'[first_reset_states] frame:{t}', fontsize=10)
            axs[1].set_title(f'[second_reset_states] frame:{t}', fontsize=10)
            if np.allclose(first_reset_states[t], second_reset_states[t]) == False:
                print(f'frames at step: {t} are not the same!')
            else:
                print(f'frames at step: {t} are the same!')

            plt.pause(1/60)

    # Check that they played different games btw resets:
    for t in range(len(first_reset_states)):
        if not np.allclose(first_reset_states[t], second_reset_states[t]):
            print(f'frames at step: {t} are different!')
            print('[Test] "test_after_reset_the_same_env_plays_different_games": PASSED')
            return True

    print('[Test] "test_after_reset_the_same_env_plays_different_games": FAILED')
    return False


def test_when_initialized_with_the_same_seed_the_envs_play_the_same_game(visualize=False):
    """
    Check that initializing two envs from scratch with the same seeds would play the same game given that we take the same actions in both of them.
    """
    # Set seed:
    from utils import set_seed
    seed = 42
    # First env episode collection:
    set_seed(seed) # Note: this will not seed the gym environment

    # Create parallelized envs:
    from dist_utils import create_parallel_env_processes
    from envs import AtariEnvironment
    num_env_per_process = 1
    env_type = AtariEnvironment
    env_id = 'PongNoFrameskip-v4'
    sticky_action = True
    action_prob = 0.25
    life_done = False
    stateStackSize = 4
    input_size = 84
    output_size = 6
    class DummyLogger:
        def __init__(self):
            pass
        def log_msg_to_both_console_and_file(self, *args, **kwargs):
            # print(*args, **kwargs)
            pass
    logger = DummyLogger()
    _, env_worker_parent_conns, _ = create_parallel_env_processes(num_env_per_process, env_type, env_id, sticky_action, action_prob, life_done, stateStackSize, input_size, seed, logger)

    first_env_ep_done_flag = False
    first_env_actions = []
    first_env_states = []
    while not first_env_ep_done_flag:
        # Send actions:
        actions = np.random.randint(0, output_size, size=(num_env_per_process,)).astype(dtype=np.int64) # Note that random action taking might be an issue which results in same images being used for training
        for parent_conn, action in zip(env_worker_parent_conns, actions):
            parent_conn.send(action)
            first_env_actions.append(actions)

        # Get observations:
        next_states = np.zeros([num_env_per_process, stateStackSize, input_size, input_size], dtype=np.float64) # -> [num_env, state_stack_size, H, W]
        for env_idx, parent_conn in enumerate(env_worker_parent_conns):
            s, r, d, trun, visited_rooms = parent_conn.recv()
            next_states[env_idx] = s[:]
            first_env_states.append(s)

            if d or trun:
                info = {'episode': {}}
                if 'Montezuma' in env_id:
                    info['episode']['number_of_visited_rooms'], info['episode']['visited_rooms'] = parent_conn.recv()
                    number_of_visited_rooms.append(info['episode']['number_of_visited_rooms'])
                info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes'] = parent_conn.recv()

                print(f'first env episode collected', flush=True)
                first_env_ep_done_flag = True
                break
        
    
    # Second env episode collection:
    set_seed(seed) # Note: this will not seed the gym environment
    _, env_worker_parent_conns, _ = create_parallel_env_processes(num_env_per_process, env_type, env_id, sticky_action, action_prob, life_done, stateStackSize, input_size, seed, logger)

    second_env_ep_done_flag = False
    second_env_states = []
    for action_idx in range(len(first_env_actions)):
        # Send actions:
        for parent_conn, action in zip(env_worker_parent_conns, first_env_actions[action_idx]):
            parent_conn.send(action)

        # Get observations:
        next_states = np.zeros([num_env_per_process, stateStackSize, input_size, input_size], dtype=np.float64) # -> [num_env, state_stack_size, H, W]
        for env_idx, parent_conn in enumerate(env_worker_parent_conns):
            s, r, d, trun, visited_rooms = parent_conn.recv()
            next_states[env_idx] = s[:]
            second_env_states.append(s)

            if d or trun:
                info = {'episode': {}}
                if 'Montezuma' in env_id:
                    info['episode']['number_of_visited_rooms'], info['episode']['visited_rooms'] = parent_conn.recv()
                    number_of_visited_rooms.append(info['episode']['number_of_visited_rooms'])
                info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes'] = parent_conn.recv()

                print(f'second env episode collected', flush=True)
                second_env_ep_done_flag = True
                break



    # Play the two recorded episodes side by side for visual debugging:
    if visualize:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        plt.ion()
        for t in range(len(first_env_actions)):
            axs[0].imshow(first_env_states[t][-1:,:].transpose(1,2,0), cmap='gray')
            axs[1].imshow(second_env_states[t][-1:,:].transpose(1,2,0), cmap='gray')


            axs[0].set_title(f'[first_env_states] frame:{t}', fontsize=10)
            axs[1].set_title(f'[second_env_states] frame:{t}', fontsize=10)
            if np.allclose(first_env_states[t], second_env_states[t]) == False:
                print(f'frames at step: {t} are not the same!')

            plt.pause(1/60)


    for t in range(len(first_env_actions)):
        if np.allclose(first_env_states[t], second_env_states[t]) == False:
                print(f'frames at step: {t} are not the same !')
                breakpoint()
        assert np.allclose(first_env_states[t], second_env_states[t]), f'frames at times step: {t} are not the same !'
    # Check that both envs have played the same episode given that the same actions were taken:
    assert first_env_ep_done_flag and second_env_ep_done_flag and np.allclose(first_env_states[-2], second_env_states[-2]), "reproducing the same env did not play out the same way given the same actions"

    print('[Test] "test_single_env_reset_reproducibility_seeds": PASSED')
    return True


def test_obs_rms():
    """
    Check that RunningMeanStd fn works as intended as obs_rms.
    """
    from utils import RunningMeanStd

    for train_method in ['original_RND', 'modifiedRND']:
        input_size = 84
        extracted_feature_embedding_dim = 448

        if train_method == 'original_RND':
            obs_rms = RunningMeanStd(shape=(1, 1, input_size, input_size), usage='obs_rms') # used for normalizing inputs to RND module (i.e. extracted_feature_embeddings)
        elif train_method == 'modified_RND':
            obs_rms = RunningMeanStd(shape=(1, extracted_feature_embedding_dim), usage='obs_rms') # used for normalizing inputs to RND module (i.e. extracted_feature_embeddings)

        # dummy inputs
        num_step = 16
        num_env_workers = 3
        if train_method == 'original_RND':
            next_obs = np.ones((num_step*num_env_workers, 1, input_size, input_size))
        elif train_method == 'modified_RND':
            extracted_feature_embeddings = np.ones((num_step*num_env_workers, extracted_feature_embedding_dim))

        if train_method == 'original_RND':
            assert (list(next_obs.shape) == [num_step*num_env_workers, 1, input_size, input_size])
            obs_rms.update(next_obs)
        elif train_method == 'modified_RND':
            with torch.no_grad(): # gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                assert (list(extracted_feature_embeddings.shape) == [num_step*num_env_workers, extracted_feature_embedding_dim])
            obs_rms.update(extracted_feature_embeddings)
        
        assert np.allclose(obs_rms.mean, 1.)
    
    print(f'[Test] "test_obs_rms": PASSED')
    return True


if __name__ == "__main__":
    # Test Wrappers:
    # test_MaxStepPerEpisodeWrapper()
    # test_MaxAndSkipEnvWrapper(render=False)
    # test_MontezumaInfoWrapper()
    # test_FrameStackWrapper()
    # test_StickyActionWrapper()
    # test_MonitorWrapper()
    # test_ResizeAndGrayScaleWrapper()
    # test_RGBArrayAsObservationWrapper()
    
    # Test Custom Parallelized Environments:
    # test_CustomEnvironments(env_type='atari', env_id='MontezumaRevengeNoFrameskip-v4') # test AtariEnvironment
    # test_CustomEnvironments(env_type='mario', env_id='SuperMarioBros-v0') # test MarioEnvironment
    # test_CustomEnvironments(env_type='mario', env_id='SuperMarioBrosRandomStages-v0') # test MarioEnvironment
    # test_CustomEnvironments(env_type='classic_control', env_id='CartPole-v1') # test ClassicControlEnvironment

    # ATARI ENV tests:
    is_passed1 = test_every_paralell_env_is_playing_different_games(visualize=False, take_same_action=True)
    is_passed2 = test_after_reset_the_same_env_plays_different_games(visualize=False) 
    is_passed3 = test_when_initialized_with_the_same_seed_the_envs_play_the_same_game(visualize=False) 

    # Utility fn tests:
    is_passed4 = test_obs_rms()

    print('='*100)
    print(f'TESTS PASSED = {int(is_passed1)+int(is_passed2)+int(is_passed3)+int(is_passed4)}/{4}')
    print('='*100)

    # test_Atari()