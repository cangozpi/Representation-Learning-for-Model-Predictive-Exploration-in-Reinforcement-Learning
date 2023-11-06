from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

import numpy as np
from utils import Logger, set_seed
from os import path
from collections import deque
from dist_utils import ddp_setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# from torch.distributed.elastic.multiprocessing.errors import record
# @record
def main(args):
    logger = Logger(file_log_path=path.join("logs", "file_logs", args['log_name']), tb_log_path=path.join("logs", "tb_logs", args['log_name']))

    use_cuda = default_config.getboolean('UseGPU')
    GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK,\
         gpu_id,\
             agents_group, env_workers_group,\
                 agents_group_global_ranks, env_workers_group_global_ranks, env_workers_group_per_node_global_ranks = ddp_setup(logger, use_cuda)
    
    dist.barrier() # wait for process initialization logging inside ddp_setup() to finish
    
    dist_tags = {
        'action': 0,
        'state': 1,
        'reward': 2,
        'done': 3,
        'truncated': 4,
        'number_of_visited_rooms': 5,
        'undiscounted_episode_return': 6,
        'episode_length': 7
    }

    logger.GLOBAL_RANK = GLOBAL_RANK
    logger.log_msg_to_both_console_and_file(
        "*" * 30 + "\n" +
        str(dict(**{section: dict(config[section]) for section in config.sections()}, **args)) + "\n"
        + f'total number of agent workers: {len(agents_group_global_ranks)}, total number of environment workers: {len(env_workers_group_global_ranks)}, number of agent workers per node: {1}, number of environment workers per node: {len(env_workers_group_per_node_global_ranks)}'
        + "\n" + "*" * 30,
        only_rank_0=True
        )


    seed = args['seed'] + GLOBAL_RANK # set different seed to every env_worker process so that every env does not play the same game
    set_seed(seed) # Note: this will not seed the gym environment


    train_method = default_config['TrainMethod']
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

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = default_config.getboolean('loadModel')
    is_render = default_config.getboolean('render')
    load_ckpt_path = '{}'.format(args['load_model_path']) # path for resuming a training from a checkpoint
    save_ckpt_path = '{}'.format(args['save_model_path']) # path for saving a training from to a checkpoint

    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    GAE_Lambda = float(default_config['GAELambda'])
    num_env_workers = len(env_workers_group_per_node_global_ranks)

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

    reward_rms = RunningMeanStd() # used for normalizing intrinsic rewards
    # obs_rms = RunningMeanStd(shape=(1, 1, input_size, input_size)) # used for normalizing observations
    extracted_feature_embedding_dim = 32 # TODO: set this automatically by calculation
    obs_rms = RunningMeanStd(shape=(1, extracted_feature_embedding_dim)) # used for normalizing observations
    pre_obs_norm_step = int(default_config['ObsNormStep'])
    discounted_reward = RewardForwardFilter(int_gamma) # gamma used for calculating Returns for the intrinsic rewards (i.e. R_i)

    agent = RNDAgent

    if default_config['EnvType'] == 'atari':
        env_type = AtariEnvironment
    elif default_config['EnvType'] == 'mario':
        env_type = MarioEnvironment
    elif default_config['EnvType'] == 'classic_control':
        env_type = ClassicControlEnvironment
    else:
        raise NotImplementedError

    

    if GLOBAL_RANK in agents_group_global_ranks: # Agent/Trainer Processes

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
                # agent.model.load_state_dict(load_checkpoint['agent.model.state_dict'])
                # agent.rnd.predictor.load_state_dict(load_checkpoint['agent.rnd.predictor.state_dict'])
                # agent.rnd.target.load_state_dict(load_checkpoint['agent.rnd.target.state_dict'])
                if representation_lr_method == "BYOL": # BYOL
                    # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                    assert agent.representation_model.net is agent.model.feature
                    # agent.representation_model.net = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                    # assert agent.representation_model.net is agent.model.feature
                if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                    # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                    assert agent.representation_model.backbone is agent.model.feature
                    # agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                    # assert agent.representation_model.backbone is agent.model.feature
                # agent.optimizer.load_state_dict(load_checkpoint['agent.optimizer.state_dict'])

            else:
                load_checkpoint = torch.load(load_ckpt_path, map_location='cpu')
                agent.load_state_dict(load_checkpoint['agent.state_dict'])
                # agent.model.load_state_dict(load_checkpoint['agent.model.state_dict'])
                # agent.rnd.predictor.load_state_dict(load_checkpoint['agent.rnd.predictor.state_dict'])
                # agent.rnd.target.load_state_dict(load_checkpoint['agent.rnd.target.state_dict'])
                if representation_lr_method == "BYOL": # BYOL
                    # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                    assert agent.representation_model.net is agent.model.feature
                    # agent.representation_model.net = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                    # assert agent.representation_model.net is agent.model.feature
                if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                    # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                    assert agent.representation_model.backbone is agent.model.feature
                    # agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                    # assert agent.representation_model.backbone is agent.model.feature
                # agent.optimizer.load_state_dict(load_checkpoint['agent.optimizer.state_dict'])
            
            
            for param in agent.rnd.target.parameters():
                assert param.requires_grad == False

            obs_rms = load_checkpoint['obs_rms']
            reward_rms = load_checkpoint['reward_rms']
            discounted_reward = load_checkpoint['discounted_reward']
            global_update = load_checkpoint['global_update']
            global_step = load_checkpoint['global_step']
            undiscounted_episode_return = load_checkpoint['undiscounted_episode_return']
            episode_lengths = load_checkpoint['episode_lengths']
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
            process_group=agents_group
            )

    
        agent_PPO_total_params = sum(p.numel() for p in agent.module.model.parameters())
        agent_RND_predictor_total_params = sum(p.numel() for p in agent.module.rnd.predictor.parameters())
        agent_representation_model_total_params = sum(p.numel() for p in agent.module.representation_model.parameters()) if agent.module.representation_model is not None else 0
        logger.log_msg_to_both_console_and_file(f"{'*'*20}\
            \nNumber of PPO parameters: {agent_PPO_total_params}\
            \nNumber of RND_predictor parameters: {agent_RND_predictor_total_params}\
            \nNumber of {representation_lr_method if representation_lr_method != 'None' else 'Representation Learning Model'} parameters: {agent_representation_model_total_params}\
            \n{'*'*20}", only_rank_0=True)

        agent.module.set_mode("train")


        states = np.zeros([num_env_workers, stateStackSize, input_size, input_size])
        
        NUM_LOCAL_ENV_WORKERS = LOCAL_WORLD_SIZE - 1 # number of env worker processes running on the current node
        LOCAL_ENV_WORKER_GLOBAL_RANKS = [GLOBAL_RANK + env_worker_local_rank for env_worker_local_rank in range(1, (NUM_LOCAL_ENV_WORKERS + 1))] # global ranks of the env workers which are working on the current node

        # normalize obs
        if is_load_model == False:
            logger.log_msg_to_both_console_and_file('Start to initialize observation normalization parameter.....', only_rank_0=True)
            if is_render:
                renderer = ParallelizedEnvironmentRenderer(num_env_workers)
            next_obs = []
            for step in range(num_step * pre_obs_norm_step):
                actions = torch.tensor(np.random.randint(0, output_size, size=(num_env_workers,)), dtype=torch.int64)

                for local_env_worker_global_rank, action in zip(LOCAL_ENV_WORKER_GLOBAL_RANKS, actions):
                    dist.send(action, dst=local_env_worker_global_rank, tag=dist_tags['action'])
                
                for local_env_worker_global_rank in LOCAL_ENV_WORKER_GLOBAL_RANKS:
                    s, r, d, trun = torch.zeros(stateStackSize, input_size, input_size, dtype=torch.uint8), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
                    dist.recv(s, src=local_env_worker_global_rank, tag=dist_tags['state'])
                    dist.recv(r, src=local_env_worker_global_rank, tag=dist_tags['reward'])
                    dist.recv(d, src=local_env_worker_global_rank, tag=dist_tags['done'])
                    dist.recv(trun, src=local_env_worker_global_rank, tag=dist_tags['truncated'])

                    if d or trun:
                        info = {'episode': {}}
                        info['episode']['undiscounted_episode_return'] = torch.zeros(1, dtype=torch.float64)
                        info['episode']['l'] = torch.zeros(1, dtype=torch.float64)
                        if 'Montezuma' in env_id:
                            info['episode']['number_of_visited_rooms'] = torch.zeros(1, dtype=torch.float64)
                            dist.recv(info['episode']['number_of_visited_rooms'], src=local_env_worker_global_rank, tag=dist_tags['number_of_visited_rooms'])
                        dist.recv(info['episode']['undiscounted_episode_return'], src=local_env_worker_global_rank, tag=dist_tags['undiscounted_episode_return'])
                        dist.recv(info['episode']['l'], src=local_env_worker_global_rank, tag=dist_tags['episode_length'])

                    # next_obs.append(s[stateStackSize - 1, :, :].reshape([1, input_size, input_size]))
                    next_obs.append(s) # [stateStackSize, input_size, input_size]
                

                if is_render:
                    renderer.render(np.stack(next_obs[-num_env_workers:]))
                if len(next_obs) % (num_step * num_env_workers) == 0:
                    next_obs = np.stack(next_obs) # [(num_step * num_env_workers), stateStackSize, input_size, input_size]
                    with torch.no_grad(): # gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                        extracted_feature_embeddings = agent.module.extract_feature_embeddings(next_obs / 255).cpu().numpy() # [(num_step * num_env_workers), feature_embeddings_dim]
                    obs_rms.update(extracted_feature_embeddings)
                    # obs_rms.update(next_obs)
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
                actions, value_ext, value_int, policy = agent.module.get_action(np.float32(states) / 255.)
                actions = torch.tensor(actions, dtype=torch.int64)

                for idx, local_env_worker_global_rank in enumerate(LOCAL_ENV_WORKER_GLOBAL_RANKS):
                    dist.send(actions[idx], dst=local_env_worker_global_rank, tag=dist_tags['action'])


                next_states, rewards, dones, next_obs = [], [], [], []
                for local_env_worker_global_rank in LOCAL_ENV_WORKER_GLOBAL_RANKS:
                    s, r, d, trun = torch.zeros(stateStackSize, input_size, input_size, dtype=torch.uint8), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
                    dist.recv(s, src=local_env_worker_global_rank, tag=dist_tags['state'])
                    dist.recv(r, src=local_env_worker_global_rank, tag=dist_tags['reward'])
                    dist.recv(d, src=local_env_worker_global_rank, tag=dist_tags['done'])
                    dist.recv(trun, src=local_env_worker_global_rank, tag=dist_tags['truncated'])

                    next_states.append(s)
                    rewards.append(r)
                    dones.append(d) # --> [num_env]
                    # next_obs.append(s[(stateStackSize - 1), :, :].reshape([1, input_size, input_size]))
                    next_obs.append(s) # [stateStackSize, input_size, input_size]

                    if d or trun:
                        info = {'episode': {}}
                        info['episode']['undiscounted_episode_return'] = torch.zeros(1, dtype=torch.float64)
                        info['episode']['l'] = torch.zeros(1, dtype=torch.float64)
                        if 'Montezuma' in env_id:
                            info['episode']['number_of_visited_rooms'] = torch.zeros(1, dtype=torch.float64)
                            dist.recv(info['episode']['number_of_visited_rooms'], src=local_env_worker_global_rank, tag=dist_tags['number_of_visited_rooms'])
                            number_of_visited_rooms.append(info['episode']['number_of_visited_rooms'].item())
                        dist.recv(info['episode']['undiscounted_episode_return'], src=local_env_worker_global_rank, tag=dist_tags['undiscounted_episode_return'])
                        dist.recv(info['episode']['l'], src=local_env_worker_global_rank, tag=dist_tags['episode_length'])
                        undiscounted_episode_return.append(info['episode']['undiscounted_episode_return'].item())
                        episode_lengths.append(info['episode']['l'].item())


                next_states = np.stack(next_states) # -> [num_env, state_stack_size, H, W]
                rewards = np.hstack(rewards) # -> [num_env, ]
                dones = np.hstack(dones) # -> [num_env, ]
                # next_obs = np.stack(next_obs) # -> [num_env, 1, H, W]
                next_obs = np.stack(next_obs) # -> [num_env, stateStackSize, H, W]

                # Compute normalize obs, compute intrinsic rewards and clip them (note that: total reward = int reward + ext reward)
                with torch.no_grad(): # gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                    extracted_feature_embeddings = agent.module.extract_feature_embeddings(next_obs / 255).cpu().numpy() # [num_worker_envs, feature_embeddings_dim]
                    intrinsic_reward = agent.module.compute_intrinsic_reward(
                        ((extracted_feature_embeddings - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]

                # intrinsic_reward = agent.module.compute_intrinsic_reward(
                #     ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]
                intrinsic_reward = np.hstack(intrinsic_reward) # [num_env,]

                total_next_obs.append(next_obs) # --> [num_step, num_env, state_stack_size, H, W]
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
                    renderer.render(next_obs)


            # calculate last next value
            _, value_ext, value_int, _ = agent.module.get_action(np.float32(states) / 255.)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            # --------------------------------------------------

            total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, stateStackSize, input_size, input_size]) # --> [num_env * num_step, state_stack_size, H, W]
            total_reward = np.stack(total_reward).transpose().clip(-1, 1) # --> [num_env, num_step]
            total_action = np.stack(total_action).transpose().reshape([-1]) # --> [num_env * num_step]
            total_done = np.stack(total_done).transpose() # --> [num_env, num_step]
            # total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, input_size, input_size]) # --> [num_env * num_step, 1, H, W]
            total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, stateStackSize, input_size, input_size]) # --> [num_env * num_step, state_stack_size, H, W]
            total_ext_values = np.stack(total_ext_values).transpose() # --> [num_env, (num_step + 1)]
            total_int_values = np.stack(total_int_values).transpose() # --> [num_env, (num_step + 1)]


            # Step 2. calculate intrinsic reward
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
            # None Episodic (hence the np.zeros() for the done input)
            int_target, int_adv = make_train_data(total_int_reward,
                                                np.zeros_like(total_int_reward),
                                                total_int_values,
                                                int_gamma,
                                                num_step,
                                                num_env_workers)

            # add ext adv and int adv
            total_adv = int_adv * int_coef + ext_adv * ext_coef
            # -----------------------------------------------

            # Step 4. update obs normalize param
            # obs_rms.update(total_next_obs)

            # next_obs = np.stack(next_obs) # [(num_step * num_env_workers), stateStackSize, input_size, input_size]
            with torch.no_grad(): # gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                extracted_feature_embeddings = agent.module.extract_feature_embeddings(total_next_obs / 255).cpu().numpy() # [(num_step * num_env_workers), feature_embeddings_dim]
            obs_rms.update(extracted_feature_embeddings)
            # -----------------------------------------------

            # Step 5. Training!
            logger.log_msg_to_both_console_and_file(f'[RANK:{GLOBAL_RANK} | {gpu_id}] global_step: {global_step}, global_update: {global_update} | ENTERED TRAINING:')
            # agent.module.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
            #                 total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
            #                 total_policy, global_update)
            agent.module.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                            total_adv, ((extracted_feature_embeddings - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                            total_policy, global_update)
            logger.log_msg_to_both_console_and_file(f'[RANK:{GLOBAL_RANK} | {gpu_id}] global_step: {global_step}, global_update: {global_update} | EXITTED TRAINING')


            # Logging
            logger.log_scalar_to_tb_with_step('data/Mean of rollout rewards (extrinsic) vs Parameter updates', np.mean(total_reward), global_update, only_rank_0=True)
            logger.log_scalar_to_tb_with_step('data/Mean of rollout rewards (intrinsic) vs Parameter updates', np.mean(total_int_reward), global_update, only_rank_0=True)
            if len(episode_lengths) > 0: # check if any episode has been completed yet
                logger.log_scalar_to_tb_with_step('data/Mean undiscounted episodic return (over last 100 episodes) (extrinsic) vs Parameter updates', np.mean(undiscounted_episode_return), global_update, only_rank_0=True)
                logger.log_scalar_to_tb_with_step('data/Mean episode lengths (over last 100 episodes) vs Parameter updates', np.mean(episode_lengths), global_update, only_rank_0=True)
                if 'Montezuma' in env_id:
                    logger.log_scalar_to_tb_with_step('data/Mean number of rooms found (over last 100 episodes) vs Parameter updates', np.mean(number_of_visited_rooms), global_update, only_rank_0=True)


            # Save checkpoint
            if global_step % (num_env_workers * num_step * int(default_config["saveCkptEvery"])) == 0 and GLOBAL_RANK == 0:
                ckpt_dict = {
                    **{
                        'agent.state_dict': agent.module.state_dict()
                    },
                    **{
                        'agent.optimizer.state_dict': agent.module.optimizer.state_dict(),
                        'agent.model.state_dict': agent.module.model.state_dict(),
                        'agent.rnd.predictor.state_dict': agent.module.rnd.predictor.state_dict(),
                        'agent.rnd.target.state_dict': agent.module.rnd.target.state_dict(),
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
                    }),
                    **({'logger.tb_global_steps': logger.tb_global_steps})
                }
                if 'Montezuma' in env_id:
                    ckpt_dict.update(visited_rooms=number_of_visited_rooms)
                os.makedirs('/'.join(save_ckpt_path.split('/')[:-1]), exist_ok=True)
                torch.save(ckpt_dict, save_ckpt_path)
                logger.log_msg_to_both_console_and_file('Saved ckpt at Global Step :{}'.format(global_step))
            
            dist.barrier(group=agents_group)

            logger.step_pytorch_profiler(pytorch_profiler_log_path) # pytorch profiler
            logger.check_scalene_profiler_finished() # scalene profiler

    
        if is_render:
            renderer.close()

    else: # Env Worker Processes
        env_worker = env_type(env_id, False, GLOBAL_RANK, dist_tags=dist_tags, sticky_action=sticky_action, p=action_prob,
                            life_done=life_done, history_size=stateStackSize, seed=seed, logger=logger) # Note that seed+rank is required to make parallel envs play different scenarios
        env_worker.run()
