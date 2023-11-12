from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

import numpy as np
from utils import Logger, set_seed
from os import path
from collections import deque
from dist_utils import ddp_setup, create_parallel_env_processes
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# from torch.distributed.elastic.multiprocessing.errors import record
# @record
def main(args):
    logger = Logger(file_log_path=path.join("logs", "file_logs", args['log_name']), tb_log_path=path.join("logs", "tb_logs", args['log_name']))

    use_cuda = default_config.getboolean('UseGPU')
    GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK, gpu_id = ddp_setup(logger, use_cuda)
    
    dist.barrier() # wait for process initialization logging inside ddp_setup() to finish


    logger.GLOBAL_RANK = GLOBAL_RANK
    # logger.log_msg_to_both_console_and_file(
    #     "*" * 30 + "\n" +
    #     str(dict(**{section: dict(config[section]) for section in config.sections()}, **args)) + "\n"
    #     + f'total number of agent workers: {len(agents_group_global_ranks)}, total number of environment workers: {len(env_workers_group_global_ranks)}, number of agent workers per node: {1}, number of environment workers per node: {len(env_workers_group_per_node_global_ranks)}'
    #     + "\n" + "*" * 30,
    #     only_rank_0=True
    #     )

    num_env_per_process = int(args['num_env_per_process'])
    assert num_env_per_process > 1, "num_env_per_process has to be larger than 1"
    num_env_workers = num_env_per_process

    seed = args['seed'] + (GLOBAL_RANK * num_env_workers) # set different seed to every env_worker process so that every env does not play the same game
    set_seed(seed) # Note: this will not seed the gym environment


    train_method = default_config['TrainMethod']
    assert train_method in ['PPO', 'original_RND', 'modified_RND']
    representation_lr_method = str(default_config['representationLearningMethod'])

    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        from nes_py.wrappers import JoypadSpace
        # env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
        env = JoypadSpace(gym_super_mario_bros.make(env_id, apply_api_compatibility=True), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    elif env_type == 'classic_control':
        env = gym.make(env_id)
    else:
        raise NotImplementedError

    if default_config['PreProcHeight'] is not None:
        input_size = int(default_config['PreProcHeight'])
    else:
        input_size = env.observation_space.shape  # 4

    if isinstance(env.action_space, gym.spaces.box.Box):
        output_size = env.action_space.shape[0]
    else:
        output_size = env.action_space.n  # 2

    if 'Breakout' in env_id: #TODO: not sure why this was done in other implementations
        output_size -= 1

    env.close()

    is_load_model = default_config.getboolean('loadModel')
    is_render = default_config.getboolean('render')
    load_ckpt_path = '{}'.format(args['load_model_path']) # path for resuming a training from a checkpoint
    save_ckpt_path = '{}'.format(args['save_model_path']) # path for saving a training from to a checkpoint

    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    GAE_Lambda = float(default_config['GAELambda'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_env_workers / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    int_gamma = float(default_config['IntGamma'])
    max_grad_norm = float(default_config['MaxGradNorm'])
    ext_coef = float(default_config['ExtCoef'])
    int_coef = float(default_config['IntCoef'])

    sticky_action = default_config.getboolean('StickyAction')
    stateStackSize = int(default_config['StateStackSize'])
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

    reward_rms = RunningMeanStd(usage='reward_rms') # used for normalizing intrinsic rewards
    if train_method == 'original_RND':
        obs_rms = RunningMeanStd(shape=(1, 1, input_size, input_size), usage='obs_rms') # used for normalizing observations
    elif train_method == 'modified_RND':
        extracted_feature_embedding_dim = CnnActorCriticNetwork.extracted_feature_embedding_dim
        obs_rms = RunningMeanStd(shape=(1, extracted_feature_embedding_dim), usage='obs_rms') # used for normalizing observations
    elif train_method == 'PPO':
        obs_rms = None
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma) # gamma used for calculating Returns for the intrinsic rewards (i.e. R_i)
    highest_mean_total_reward = - float("inf")
    highest_mean_undiscounted_episode_return = - float("inf")

    agent = RNDAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    elif default_config['EnvType'] == 'classic_control':
        env_type = ClassicControlEnvironment
    else:
        raise NotImplementedError
    


    # Create environment processes
    _, env_worker_parent_conns, _ = create_parallel_env_processes(num_env_per_process, env_type, env_id, sticky_action, action_prob, life_done, stateStackSize, input_size, seed, logger)


    agent = agent(
        input_size,
        output_size,
        num_env_workers,
        num_step,
        gamma,
        GAE_Lambda=GAE_Lambda,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        representation_lr_method=representation_lr_method,
        device=gpu_id,
        logger=logger,
    )

    global_update = 0
    global_step = 0
    undiscounted_episode_return = deque([], maxlen=100)
    episode_lengths = deque([], maxlen=100)
    if 'Montezuma' in env_id:
        number_of_visited_rooms = deque([], maxlen=100)

    if is_load_model:
        logger.log_msg_to_both_console_and_file(f'loading from checkpoint: {load_ckpt_path}', only_rank_0=True)
        if use_cuda:
            load_checkpoint = torch.load(load_ckpt_path, map_location=gpu_id)
            agent.load_state_dict(load_checkpoint['agent.state_dict'])
            if representation_lr_method == "BYOL": # BYOL
                assert agent.representation_model.net is agent.model.feature
            if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                assert agent.representation_model.backbone is agent.model.feature
            # agent.optimizer.load_state_dict(load_checkpoint['agent.optimizer.state_dict'])

        else:
            load_checkpoint = torch.load(load_ckpt_path, map_location='cpu')
            agent.load_state_dict(load_checkpoint['agent.state_dict'])
            if representation_lr_method == "BYOL": # BYOL
                assert agent.representation_model.net is agent.model.feature
            if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                assert agent.representation_model.backbone is agent.model.feature
            # agent.optimizer.load_state_dict(load_checkpoint['agent.optimizer.state_dict'])
            
            
        if train_method in ['original_RND', 'modified_RND']:
            for param in agent.rnd.target.parameters():
                assert param.requires_grad == False

        obs_rms = load_checkpoint['obs_rms']
        reward_rms = load_checkpoint['reward_rms']
        discounted_reward = load_checkpoint['discounted_reward']
        global_update = load_checkpoint['global_update']
        global_step = load_checkpoint['global_step']
        undiscounted_episode_return = load_checkpoint['undiscounted_episode_return']
        episode_lengths = load_checkpoint['episode_lengths']
        highest_mean_total_reward = load_checkpoint['highest_total_reward']
        highest_mean_undiscounted_episode_return  = load_checkpoint['highest_mean_undiscounted_episode_return']
        if 'Montezuma' in env_id:
            number_of_visited_rooms = load_checkpoint['visited_rooms']
        logger.tb_global_steps = load_checkpoint['logger.tb_global_steps']

        logger.log_msg_to_both_console_and_file('loading finished!', only_rank_0=True)
        
    if "cuda" in gpu_id: # SyncBatchNorm layers only work with GPU modules
        agent = nn.SyncBatchNorm.convert_sync_batchnorm(agent, process_group=agents_group) # synchronizes batch norm stats for dist training

    agent = DDP(
        agent, 
        device_ids=None if gpu_id == "cpu" else [gpu_id], 
        output_device=None if gpu_id == "cpu" else gpu_id,
        )

    
    agent_PPO_total_params = sum(p.numel() for p in agent.module.model.parameters())
    agent_RND_predictor_total_params = sum(p.numel() for p in agent.module.rnd.predictor.parameters()) if agent.module.rnd is not None else 0
    agent_representation_model_total_params = sum(p.numel() for p in agent.module.representation_model.parameters()) if agent.module.representation_model is not None else 0
    logger.log_msg_to_both_console_and_file(f"{'*'*20}\
        \nNumber of PPO parameters: {agent_PPO_total_params}\
        \nNumber of RND_predictor parameters: {agent_RND_predictor_total_params}\
        \nNumber of {representation_lr_method if representation_lr_method != 'None' else 'Representation Learning Model'} parameters: {agent_representation_model_total_params}\
        \n{'*'*20}", only_rank_0=True)

    agent.module.set_mode("train")


    states = np.zeros([num_env_workers, stateStackSize, input_size, input_size])
        
    # normalize obs
    if is_load_model == False:
        logger.log_msg_to_both_console_and_file('Start to initialize observation normalization parameter.....', only_rank_0=True)
        if is_render:
            renderer = ParallelizedEnvironmentRenderer(num_env_workers)
        next_obs = []
        for step in range(num_step * pre_obs_norm_step):
            actions = np.random.randint(0, output_size, size=(num_env_workers,)).astype(dtype=np.int64)

            for parent_conn, action in zip(env_worker_parent_conns, actions):
                parent_conn.send(action)
                
            for env_idx, parent_conn in enumerate(env_worker_parent_conns):
                s, r, d, trun = parent_conn.recv()
                assert (list(s.shape) == [stateStackSize, input_size, input_size]) and (s.dtype == np.float64)
                assert type(r) == float
                assert type(d) == bool
                assert type(trun) == bool

                if d or trun:
                    info = {'episode': {}}
                    if 'Montezuma' in env_id:
                        info['episode']['number_of_visited_rooms'], info['episode']['visited_rooms'] = parent_conn.recv()
                    info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes'] = parent_conn.recv()
                    # Logging:
                    if 'Montezuma' in env_id:
                        logger.log_msg_to_both_console_and_file(f'[Rank: {GLOBAL_RANK}, env: {env_idx}] episode: {info["episode"]["num_finished_episodes"]}, step: {info["episode"]["l"]}, undiscounted_return: {info["episode"]["undiscounted_episode_return"]}, moving_average_undiscounted_return: {np.mean(info["episode"]["undiscounted_episode_return"])}, visited_rooms: {info["episode"]["visited_rooms"]}')
                    else:
                        logger.log_msg_to_both_console_and_file(f'[Rank: {GLOBAL_RANK}, env: {env_idx} ] episode: {info["episode"]["num_finished_episodes"]}, step: {info["episode"]["l"]}, undiscounted_return: {info["episode"]["undiscounted_episode_return"]}, moving_average_undiscounted_return: {np.mean(info["episode"]["undiscounted_episode_return"])}')

                if train_method == 'original_RND':
                    next_obs.append(s[stateStackSize - 1, :, :].reshape([1, input_size, input_size])) # [1, input_size, input_size]
                elif train_method == 'modified_RND':
                    next_obs.append(s) # [stateStackSize, input_size, input_size]
                elif (train_method == 'PPO') and is_render: # next_obs is just used for rendering purposes
                    next_obs.append(s) # [stateStackSize, input_size, input_size]
                

            if is_render:
                if train_method == 'original_RND':
                    renderer.render(np.stack(next_obs[-num_env_workers:])) # [num_env, 1, input_size, input_size]
                elif train_method == 'modified_RND':
                    renderer.render(np.stack(next_obs[-num_env_workers:])[:, -1, :, :].reshape([num_env_workers, 1, input_size, input_size])) # [num_env, 1, input_size, input_size]
                elif train_method == 'PPO':
                    renderer.render(np.stack(next_obs[-num_env_workers:])[:, -1, :, :].reshape([num_env_workers, 1, input_size, input_size])) # [num_env, 1, input_size, input_size]
            if train_method in ['original_RND', 'modified_RND']:
                if len(next_obs) % (num_step * num_env_workers) == 0:
                    next_obs = np.stack(next_obs) # modified_RND: [(num_step * num_env_workers), stateStackSize, input_size, input_size], original_RND:[(num_step * num_env_workers), 1, input_size, input_size]
                    if train_method == 'original_RND':
                        assert (list(next_obs.shape) == [num_step*num_env_workers, 1, input_size, input_size])
                        obs_rms.update(next_obs)
                    elif train_method == 'modified_RND':
                        with torch.no_grad(): # gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                            assert (list(next_obs.shape) == [num_step*num_env_workers, stateStackSize, input_size, input_size])
                            extracted_feature_embeddings = agent.module.extract_feature_embeddings(next_obs / 255).cpu().numpy() # [(num_step * num_env_workers), feature_embeddings_dim]
                            assert (list(extracted_feature_embeddings.shape) == [num_step*num_env_workers, extracted_feature_embedding_dim])
                        obs_rms.update(extracted_feature_embeddings)
                    next_obs = []
        if is_render:
            renderer.close()
        logger.log_msg_to_both_console_and_file('End to initialize...', only_rank_0=True)
        

    if is_render:
        renderer = ParallelizedEnvironmentRenderer(num_env_workers)

    # pytorch profiling:
    pytorch_profiler_log_path = f'./logs/torch_profiler_logs/{args["log_name"]}_TrainingLoop_prof_rank{GLOBAL_RANK}.log'
    logger.create_new_pytorch_profiler(pytorch_profiler_log_path, 1, 1, 3, 1)

    while True:

        total_state, total_reward, total_done, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy = \
            [], [], [], [], [], [], [], [], []
        global_step += (num_env_workers * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for step in range(num_step):
            actions, value_ext, value_int, policy = agent.module.get_action(np.float32(states) / 255.) # TODO: np.float32'yi kaldır ve assert le çöz
            assert (list(actions.shape) == [num_env_workers, ]) and (actions.dtype == np.int64)
            assert (list(value_ext.shape) == [num_env_workers, ]) and (value_ext.dtype == np.float32)
            assert (list(value_int.shape) == [num_env_workers, ]) and (value_ext.dtype == np.float32)
            assert (list(policy.shape) == [num_env_workers, output_size]) and (policy.dtype == np.float32)

            for parent_conn, action in zip(env_worker_parent_conns, actions):
                parent_conn.send(action)


            next_states, rewards, dones, next_obs = [], [], [], []
            for env_idx, parent_conn in enumerate(env_worker_parent_conns):
                s, r, d, trun = parent_conn.recv()
                assert (list(s.shape) == [stateStackSize, input_size, input_size]) and (s.dtype == np.float64)
                assert type(r) == float
                assert type(d) == bool
                assert type(trun) == bool

                next_states.append(s)
                rewards.append(r)
                dones.append(d) # --> [num_env]
                if train_method == 'original_RND':
                    next_obs.append(s[(stateStackSize - 1), :, :].reshape([1, input_size, input_size])) # [1, input_size, input_size]
                elif train_method == 'modified_RND':
                    next_obs.append(s) # [stateStackSize, input_size, input_size]

                if d or trun:
                    info = {'episode': {}}
                    if 'Montezuma' in env_id:
                        info['episode']['number_of_visited_rooms'], info['episode']['visited_rooms'] = parent_conn.recv()
                        number_of_visited_rooms.append(info['episode']['number_of_visited_rooms'])
                    info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes'] = parent_conn.recv()
                    undiscounted_episode_return.append(info['episode']['undiscounted_episode_return'])
                    episode_lengths.append(info['episode']['l'])
                    # Logging:
                    if 'Montezuma' in env_id:
                        logger.log_msg_to_both_console_and_file(f'[Rank: {GLOBAL_RANK}, env: {env_idx}] episode: {info["episode"]["num_finished_episodes"]}, step: {info["episode"]["l"]}, undiscounted_return: {info["episode"]["undiscounted_episode_return"]}, moving_average_undiscounted_return: {np.mean(info["episode"]["undiscounted_episode_return"])}, visited_rooms: {info["episode"]["visited_rooms"]}')
                    else:
                        logger.log_msg_to_both_console_and_file(f'[Rank: {GLOBAL_RANK}, env: {env_idx} ] episode: {info["episode"]["num_finished_episodes"]}, step: {info["episode"]["l"]}, undiscounted_return: {info["episode"]["undiscounted_episode_return"]}, moving_average_undiscounted_return: {np.mean(info["episode"]["undiscounted_episode_return"])}')


            next_states = np.stack(next_states) # -> [num_env, state_stack_size, H, W]
            rewards = np.hstack(rewards) # -> [num_env, ]
            dones = np.hstack(dones) # -> [num_env, ]
            assert (list(next_states.shape) == [num_env_workers, stateStackSize, input_size, input_size]) and (next_states.dtype == np.float64)
            assert (list(rewards.shape) == [num_env_workers, ]) and (rewards.dtype == np.float64)
            assert (list(dones.shape) == [num_env_workers, ]) and (dones.dtype == np.bool)
            if train_method in ['original_RND', 'modified_RND']:
                next_obs = np.stack(next_obs) # -> modified_RND: [num_env, stateStackSize, H, W], original_RND: [num_env, 1, H, W]

            # Compute normalize obs, compute intrinsic rewards and clip them (note that: total reward = int reward + ext reward)
            if train_method == 'original_RND':
                assert (list(next_obs.shape) == [num_env_workers, 1, input_size, input_size])
                intrinsic_reward = agent.module.compute_intrinsic_reward(
                    ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]
            elif train_method == 'modified_RND':
                with torch.no_grad(): # gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                    assert (list(next_obs.shape) == [num_env_workers, stateStackSize, input_size, input_size])
                    extracted_feature_embeddings = agent.module.extract_feature_embeddings(next_obs / 255).cpu().numpy() # [num_worker_envs, feature_embeddings_dim]
                    assert (list(extracted_feature_embeddings.shape) == [num_env_workers, extracted_feature_embedding_dim])
                    intrinsic_reward = agent.module.compute_intrinsic_reward(
                        ((extracted_feature_embeddings - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]

            if train_method in ['original_RND', 'modified_RND']:
                intrinsic_reward = np.hstack(intrinsic_reward) # [num_env,]

            total_next_obs.append(next_obs) # --> modified_RND: [num_step, num_env, state_stack_size, H, W], original_RND: [num_step, num_env, 1, H, W]
            if train_method in ['original_RND', 'modified_RND']:
                total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)

            states = next_states[:, :, :, :] # for an explanation of why [:, :, :, :] is used refer to the discussion: https://stackoverflow.com/questions/61103275/what-is-the-difference-between-tensor-and-tensor-in-pytorch 

            if is_render:
                if train_method == 'original_RND':
                    renderer.render(next_obs) # [num_env, 1, input_size, input_size]
                elif train_method == 'modified_RND':
                    renderer.render(next_obs[:, -1, :, :].reshape([num_env_workers, 1, input_size, input_size])) # [num_env, 1, input_size, input_size]
                elif train_method == 'PPO':
                    renderer.render(next_states[:, -1, :, :].reshape([num_env_workers, 1, input_size, input_size])) # [num_env, 1, input_size, input_size]

        # calculate last next value
        _, value_ext, value_int, _ = agent.module.get_action(np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, stateStackSize, input_size, input_size]) # --> [num_env * num_step, state_stack_size, H, W]
        total_reward = np.stack(total_reward).transpose().clip(-1, 1) # --> [num_env, num_step]
        total_action = np.stack(total_action).transpose().reshape([-1]) # --> [num_env * num_step]
        total_done = np.stack(total_done).transpose() # --> [num_env, num_step]
        if train_method == 'original_RND':
            total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, input_size, input_size]) # --> [num_env * num_step, 1, H, W]
            assert (list(total_next_obs.shape) == [num_env_workers*num_step, 1, input_size, input_size]) and (total_next_obs.dtype == np.float64)
        elif train_method == 'modified_RND':
            total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, stateStackSize, input_size, input_size]) # --> [num_env * num_step, state_stack_size, H, W]
            assert (list(total_next_obs.shape) == [num_env_workers*num_step, stateStackSize, input_size, input_size]) and (total_next_obs.dtype == np.float64)
        total_ext_values = np.stack(total_ext_values).transpose() # --> [num_env, (num_step + 1)]
        total_int_values = np.stack(total_int_values).transpose() # --> [num_env, (num_step + 1)]
        total_policy = np.stack(total_policy) # --> [num_step, num_env, output_size]
        assert (list(total_state.shape) == [num_env_workers*num_step, stateStackSize, input_size, input_size]) and (total_state.dtype == np.float64)
        assert (list(total_reward.shape) == [num_env_workers, num_step]) and (total_reward.dtype == np.float64)
        assert (list(total_action.shape) == [num_env_workers*num_step]) and (total_action.dtype == np.int64)
        assert (list(total_done.shape) == [num_env_workers, num_step]) and (total_done.dtype == np.bool)
        assert (list(total_ext_values.shape) == [num_env_workers, (num_step + 1)]) and (total_ext_values.dtype == np.float32)
        assert (list(total_int_values.shape) == [num_env_workers, (num_step + 1)]) and (total_int_values.dtype == np.float32)
        assert (list(total_policy.shape) == [num_step, num_env_workers, output_size]) and (total_policy.dtype == np.float32)


        # Step 2. calculate intrinsic reward
        if train_method in ['original_RND', 'modified_RND']:
            # running mean intrinsic reward
            total_int_reward = np.stack(total_int_reward).transpose() # --> [num_env, num_step]
            total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                            total_int_reward.T])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            reward_rms.update_from_moments(mean, std ** 2, count) # update reward normalization parameters using intrinsic rewards

            # normalize intrinsic reward
            total_int_reward /= np.sqrt(reward_rms.var)
        # -------------------------------------------------------------------------------------------

        # Step 3. make target and advantage
        # extrinsic reward calculate
        ext_target, ext_adv = make_train_data(total_reward,
                                            total_done,
                                            total_ext_values,
                                            gamma,
                                            num_step,
                                            num_env_workers)

        # intrinsic reward calculate
        if train_method in ['original_RND', 'modified_RND']:
            # None Episodic (hence the np.zeros() for the done input)
            int_target, int_adv = make_train_data(total_int_reward,
                                                np.zeros_like(total_int_reward),
                                                total_int_values,
                                                int_gamma,
                                                num_step,
                                                num_env_workers)

        # add ext adv and int adv
        if train_method in ['original_RND', 'modified_RND']:
            total_adv = int_adv * int_coef + ext_adv * ext_coef
        else:
            total_adv = ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        if train_method == 'original_RND':
            obs_rms.update(total_next_obs)
        elif train_method == 'modified_RND':
            # next_obs = np.stack(next_obs) # [(num_step * num_env_workers), stateStackSize, input_size, input_size]
            with torch.no_grad(): # gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                extracted_feature_embeddings = agent.module.extract_feature_embeddings(total_next_obs / 255).cpu().numpy() # [(num_step * num_env_workers), feature_embeddings_dim]
            obs_rms.update(extracted_feature_embeddings)
        # -----------------------------------------------

        # Logging
        logger.log_scalar_to_tb_with_step('data/Mean of rollout rewards (extrinsic) vs Parameter updates', np.mean(total_reward), global_update, only_rank_0=True)
        if train_method in ['original_RND', 'modified_RND']:
            logger.log_scalar_to_tb_with_step('data/Mean of rollout rewards (intrinsic) vs Parameter updates', np.mean(total_int_reward), global_update, only_rank_0=True)
        if len(episode_lengths) > 0: # check if any episode has been completed yet
            logger.log_scalar_to_tb_with_step('data/Mean undiscounted episodic return (over last 100 episodes) (extrinsic) vs Parameter updates', np.mean(undiscounted_episode_return), global_update, only_rank_0=True)
            logger.log_scalar_to_tb_with_step('data/Mean episode lengths (over last 100 episodes) vs Parameter updates', np.mean(episode_lengths), global_update, only_rank_0=True)
            if 'Montezuma' in env_id:
                logger.log_scalar_to_tb_with_step('data/Mean number of rooms found (over last 100 episodes) vs Parameter updates', np.mean(number_of_visited_rooms), global_update, only_rank_0=True)



        # Save checkpoint
        if (
            (global_step % (num_env_workers * num_step * int(default_config["saveCkptEvery"])) == 0) # scheduled checkpointing time
            or
            (highest_mean_total_reward < np.mean(total_reward)) # checkpointing the best performing agent so far for the metric total reward
            or
            (highest_mean_undiscounted_episode_return < np.mean(undiscounted_episode_return)) # checkpointing the best performing agent so far for the metric mean undiscounted episode return
            ) and GLOBAL_RANK == 0:

            ckpt_paths = []

            if (global_step % (num_env_workers * num_step * int(default_config["saveCkptEvery"])) == 0): # scheduled checkpointing time
                ckpt_paths.append(save_ckpt_path)

            if highest_mean_total_reward < np.mean(total_reward): # checkpointing the best performing agent so far for the metric total reward
                ckpt_path = ''.join([*save_ckpt_path.split('.')[:-1], "__BestModelForMeanExtrinsicRolloutRewards", '.' ,*save_ckpt_path.split('.')[-1:]])
                logger.log_msg_to_both_console_and_file(f'New high score for mean of rollout rewards (extrinsic): {np.mean(total_reward)}, saving checkpoint: {ckpt_path}', only_rank_0=True)
                highest_mean_total_reward = np.mean(total_reward)
                ckpt_paths.append(ckpt_path)
                
            if highest_mean_undiscounted_episode_return < np.mean(undiscounted_episode_return): # checkpointing the best performing agent so far for the metric mean undiscounted episode return
                ckpt_path = ''.join([*save_ckpt_path.split('.')[:-1], "__BestModelForMeanUndiscountedEpisodeReturn", '.' ,*save_ckpt_path.split('.')[-1:]])
                logger.log_msg_to_both_console_and_file(f'New high score for mean undiscounted episodic return (over last 100 episodes) (extrinsic): {np.mean(undiscounted_episode_return)}, saving checkpoint: {ckpt_path}', only_rank_0=True)
                highest_mean_undiscounted_episode_return = np.mean(undiscounted_episode_return)
                ckpt_paths.append(ckpt_path)


            ckpt_dict = {
                **{
                    'agent.state_dict': agent.module.state_dict()
                },
                **{
                    'agent.optimizer.state_dict': agent.module.optimizer.state_dict(),
                    'agent.model.state_dict': agent.module.model.state_dict(),
                    'agent.rnd.predictor.state_dict': agent.module.rnd.predictor.state_dict() if agent.module.rnd is not None else None,
                    'agent.rnd.target.state_dict': agent.module.rnd.target.state_dict() if agent.module.rnd is not None else None,
                },
                **({'agent.representation_model.state_dict': agent.module.representation_model.state_dict()} if representation_lr_method == "BYOL" else {}),
                **({'agent.representation_model.state_dict': agent.module.representation_model.state_dict()} if representation_lr_method == "Barlow-Twins" else {}),
                **{
                    'obs_rms': obs_rms,
                    'reward_rms': reward_rms,
                    'discounted_reward': discounted_reward,
                },
                **({
                    'global_update': global_update,
                    'global_step': global_step,
                    'undiscounted_episode_return': undiscounted_episode_return,
                    'episode_lengths': episode_lengths,
                    'highest_total_reward': highest_mean_total_reward,
                    'highest_mean_undiscounted_episode_return': highest_mean_undiscounted_episode_return,
                }),
                **({'logger.tb_global_steps': logger.tb_global_steps})
            }
            if 'Montezuma' in env_id:
                ckpt_dict.update(visited_rooms=number_of_visited_rooms)
                
            for p in ckpt_paths:
                os.makedirs('/'.join(p.split('/')[:-1]), exist_ok=True)
                torch.save(ckpt_dict, p)
                logger.log_msg_to_both_console_and_file(f'Saved ckpt: {p} at Global Step: {global_step}')
            

        # Step 5. Training!
        logger.log_msg_to_both_console_and_file(f'[RANK:{GLOBAL_RANK} | {gpu_id}] global_step: {global_step}, global_update: {global_update} | ENTERED TRAINING:')
        if train_method == 'original_RND':
            agent.module.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                            total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                            total_policy, global_update)
        elif train_method == 'modified_RND':
            agent.module.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                            total_adv, ((extracted_feature_embeddings - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                            total_policy, global_update)
        elif train_method == 'PPO':
            agent.module.train_model(np.float32(total_state) / 255., ext_target, None, total_action,
                            total_adv, None,
                            total_policy, global_update)
        logger.log_msg_to_both_console_and_file(f'[RANK:{GLOBAL_RANK} | {gpu_id}] global_step: {global_step}, global_update: {global_update} | EXITTED TRAINING')
            
        dist.barrier()

        logger.step_pytorch_profiler(pytorch_profiler_log_path) # pytorch profiler
        logger.check_scalene_profiler_finished() # scalene profiler

    
    if is_render:
        renderer.close()
