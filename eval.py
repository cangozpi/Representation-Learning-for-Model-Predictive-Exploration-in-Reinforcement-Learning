from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe


import numpy as np
import pickle
from utils import Logger, set_seed
from os import path

from dist_utils import ddp_setup
import torch.distributed as dist

def main(args):
    logger = Logger(file_log_path=path.join("logs", "file_logs", args['log_name']), tb_log_path=path.join("logs", "tb_logs", args['log_name']))
    use_cuda = default_config.getboolean('UseGPU')

    GLOBAL_WORLD_SIZE, GLOBAL_RANK, LOCAL_WORLD_SIZE, LOCAL_RANK,\
         gpu_id,\
             agents_group, env_workers_group,\
                 agents_group_global_ranks, env_workers_group_global_ranks, env_workers_group_per_node_global_ranks = ddp_setup(logger, use_cuda)
    
    assert GLOBAL_WORLD_SIZE == 2 and LOCAL_WORLD_SIZE == 2, "There should only be 2 processes and 1 node during evaluation (1 belonging to trainers_group, and 1 belonging to env_workers_group) !"

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

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = True
    is_render = True
    load_ckpt_path = '{}'.format(args['load_model_path']) # path for resuming a training from a checkpoint

    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    GAE_Lambda = float(default_config['GAELambda'])
    num_env_workers = len(env_workers_group_per_node_global_ranks)
    assert num_env_workers == 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_env_workers / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    max_grad_norm = float(default_config['MaxGradNorm'])

    sticky_action = False
    stateStackSize = int(default_config['StateStackSize'])
    action_prob = float(default_config['ActionProb'])
    life_done = default_config.getboolean('LifeDone')

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
            logger=logger
        )

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
                if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                    # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                    # agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                    assert agent.representation_model.backbone is agent.model.feature

            else:
                load_checkpoint = torch.load(load_ckpt_path, map_location='cpu')
                agent.load_state_dict(load_checkpoint['agent.state_dict'])
                # agent.model.load_state_dict(load_checkpoint['agent.model.state_dict'])
                # agent.rnd.predictor.load_state_dict(load_checkpoint['agent.rnd.predictor.state_dict'])
                # agent.rnd.target.load_state_dict(load_checkpoint['agent.rnd.target.state_dict'])
                if representation_lr_method == "BYOL": # BYOL
                    # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                    # agent.representation_model.net = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                    assert agent.representation_model.net is agent.model.feature
                if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                    # agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                    # agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
                    assert agent.representation_model.backbone is agent.model.feature


            if train_method in ['original_RND', 'modified_RND']:
                for param in agent.rnd.target.parameters():
                    assert param.requires_grad == False

            obs_rms = load_checkpoint['obs_rms']
            reward_rms = load_checkpoint['reward_rms']

            logger.log_msg_to_both_console_and_file('loading finished!', only_rank_0=True)


        agent_PPO_total_params = sum(p.numel() for p in agent.model.parameters())
        agent_RND_predictor_total_params = sum(p.numel() for p in agent.rnd.predictor.parameters()) if agent.rnd is not None else 0
        agent_representation_model_total_params = sum(p.numel() for p in agent.representation_model.parameters()) if agent.representation_model is not None else 0
        logger.log_msg_to_both_console_and_file(f"{'*'*20}\
            \nNumber of PPO parameters: {agent_PPO_total_params}\
            \nNumber of RND_predictor parameters: {agent_RND_predictor_total_params}\
            \nNumber of {representation_lr_method if representation_lr_method != 'None' else 'Representation Learning Model'} parameters: {agent_representation_model_total_params}\
            \n{'*'*20}", only_rank_0=True)

        agent.set_mode("eval")


        state = np.zeros([num_env_workers, stateStackSize, input_size, input_size]) # [num_env_workers, stateStackSize, input_size, input_size]

        NUM_LOCAL_ENV_WORKERS = LOCAL_WORLD_SIZE - 1 # number of env worker processes running on the current node
        LOCAL_ENV_WORKER_GLOBAL_RANKS = [GLOBAL_RANK + env_worker_local_rank for env_worker_local_rank in range(1, (NUM_LOCAL_ENV_WORKERS + 1))] # global ranks of the env workers which are working on the current node
        assert NUM_LOCAL_ENV_WORKERS == 1 and LOCAL_ENV_WORKER_GLOBAL_RANKS == [1]

        intrinsic_reward_list = []
        if is_render:
            renderer = ParallelizedEnvironmentRenderer(num_env_workers)
        while True:
            global_step += 1
            with torch.no_grad():
                actions, value_ext, value_int, policy = agent.get_action(np.float32(state) / 255.)
                actions = torch.tensor(actions, dtype=torch.int64)

            for idx, local_env_worker_global_rank in enumerate(LOCAL_ENV_WORKER_GLOBAL_RANKS):
                dist.send(actions[idx], dst=local_env_worker_global_rank, tag=dist_tags['action'])

            for local_env_worker_global_rank in LOCAL_ENV_WORKER_GLOBAL_RANKS:
                next_state, reward, done, trun = torch.zeros(stateStackSize, input_size, input_size, dtype=torch.uint8), torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
                dist.recv(next_state, src=local_env_worker_global_rank, tag=dist_tags['state'])
                dist.recv(reward, src=local_env_worker_global_rank, tag=dist_tags['reward'])
                dist.recv(done, src=local_env_worker_global_rank, tag=dist_tags['done'])
                dist.recv(trun, src=local_env_worker_global_rank, tag=dist_tags['truncated'])

                if train_method == 'original_RND':
                    next_obs = torch.unsqueeze(next_state[(stateStackSize - 1), :, :].reshape([1, input_size, input_size]), dim=0).type(torch.float) # [num_env_worker=1, 1, input_size, input_size]
                elif train_method == 'modified_RND':
                    next_obs = torch.unsqueeze(next_state, dim=0).type(torch.float) # [num_env_worker=1, stateStackSize, input_size, input_size]
                elif (train_method == 'PPO') and is_render: # next_obs is just used for rendering purposes
                    next_obs = torch.unsqueeze(next_state, dim=0).type(torch.float) # [num_env_worker=1, stateStackSize, input_size, input_size]

                if done or trun:
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


            # Compute normalize obs, compute intrinsic rewards and clip them (note that: total reward = int reward + ext reward)
            if train_method == 'original_RND':
                intrinsic_reward = agent.compute_intrinsic_reward(
                    ((next_obs.cpu().numpy() - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]
            elif train_method == 'modified_RND':
                with torch.no_grad(): # gradients should not flow backwards from RND to the PPO's bacbone (i.e. RND gradients should stop at the feature embeddings extracted by the PPO's bacbone)
                    extracted_feature_embeddings = agent.extract_feature_embeddings(next_obs / 255).cpu().numpy() # [num_worker_envs=1, feature_embeddings_dim]
                    intrinsic_reward = agent.compute_intrinsic_reward(
                        ((extracted_feature_embeddings - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]
            # normalize intrinsic reward
            if train_method in ['original_RND', 'modified_RND']:
                intrinsic_reward /= np.sqrt(reward_rms.var)
                intrinsic_reward_list.append(intrinsic_reward.item())

            state = torch.unsqueeze(next_state, dim=0)
                
            # if done:
            #     intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
            #         intrinsic_reward_list)
            #     with open('int_reward', 'wb') as f:
            #         pickle.dump(intrinsic_reward_list, f)


            if is_render:
                # assert list(next_obs[-num_env_workers:, -1, :, :].unsqueeze(dim=1).shape) == [num_env_workers, 1, input_size, input_size]
                # renderer.render(next_obs[-num_env_workers:, -1, :, :].unsqueeze(dim=1).numpy()) # [num_env_workers=1, 1, input_size, input_size]
                if train_method == 'original_RND':
                    renderer.render(next_obs.numpy()) # [num_env, 1, input_size, input_size]
                elif train_method == 'modified_RND':
                    renderer.render(next_obs[:, -1, :, :].reshape([num_env_workers, 1, input_size, input_size]).numpy()) # [num_env, 1, input_size, input_size]
                elif train_method == 'PPO':
                    renderer.render(next_obs[:, -1, :, :].reshape([num_env_workers, 1, input_size, input_size]).numpy()) # [num_env, 1, input_size, input_size]

        if is_render:
            renderer.close()



    else: # Env Worker Processes
        env_worker = env_type(env_id, False, GLOBAL_RANK, dist_tags=dist_tags, sticky_action=sticky_action, p=action_prob,
                            life_done=life_done, history_size=stateStackSize, seed=seed, logger=logger) # Note that seed+rank is required to make parallel envs play different scenarios
        env_worker.run()