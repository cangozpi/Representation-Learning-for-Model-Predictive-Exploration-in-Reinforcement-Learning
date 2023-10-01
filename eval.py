from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe


import numpy as np
import pickle
from utils import Logger, set_seed
from os import path


def main(args):
    set_seed(args['seed']) # Note: this will not seed the gym environment

    logger = Logger(file_log_path=path.join("logs", "file_logs", args['log_name']), tb_log_path=path.join("logs", "tb_logs", args['log_name']))
    logger.log_msg_to_both_console_and_file(str(dict(**{section: dict(config[section]) for section in config.sections()}, **args)))

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

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    GAE_Lambda = float(default_config['GAELambda'])
    num_worker = 1

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
    learning_rate = float(default_config['LearningRate'])
    entropy_coef = float(default_config['Entropy'])
    gamma = float(default_config['Gamma'])
    clip_grad_norm = float(default_config['ClipGradNorm'])

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

    agent = agent(
        input_size,
        output_size,
        num_worker,
        num_step,
        gamma,
        GAE_Lambda=GAE_Lambda,
        learning_rate=learning_rate,
        ent_coef=entropy_coef,
        clip_grad_norm=clip_grad_norm,
        epoch=epoch,
        batch_size=batch_size,
        ppo_eps=ppo_eps,
        use_cuda=use_cuda,
        use_gae=use_gae,
        use_noisy_net=use_noisy_net,
        representation_lr_method=representation_lr_method,
        logger=logger
    )


    if is_load_model:
        logger.log_msg_to_both_console_and_file(f'loading from checkpoint: {load_ckpt_path}')
        if use_cuda:
            load_checkpoint = torch.load(load_ckpt_path)
            agent.model.load_state_dict(load_checkpoint['agent.model.state_dict'])
            agent.rnd.predictor.load_state_dict(load_checkpoint['agent.rnd.predictor.state_dict'])
            agent.rnd.target.load_state_dict(load_checkpoint['agent.rnd.target.state_dict'])
            if representation_lr_method == "BYOL": # BYOL
                agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                agent.representation_model.net = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
            if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo

        else:
            load_checkpoint = torch.load(load_ckpt_path, map_location='cpu')
            agent.model.load_state_dict(load_checkpoint['agent.model.state_dict'])
            agent.rnd.predictor.load_state_dict(load_checkpoint['agent.rnd.predictor.state_dict'])
            agent.rnd.target.load_state_dict(load_checkpoint['agent.rnd.target.state_dict'])
            if representation_lr_method == "BYOL": # BYOL
                agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                agent.representation_model.net = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
            if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                agent.representation_model.load_state_dict(load_checkpoint['agent.representation_model.state_dict'])
                agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo

        logger.log_msg_to_both_console_and_file('loading finished!')


    agent_PPO_total_params = sum(p.numel() for p in agent.model.parameters())
    agent_RND_predictor_total_params = sum(p.numel() for p in agent.rnd.predictor.parameters())
    agent_representation_model_total_params = sum(p.numel() for p in agent.representation_model.parameters()) if agent.representation_model is not None else 0
    logger.log_msg_to_both_console_and_file(f"{'*'*20}\
        \nNumber of PPO parameters: {agent_PPO_total_params}\
        \nNumber of RND_predictor parameters: {agent_RND_predictor_total_params}\
        \nNumber of {representation_lr_method if representation_lr_method != 'None' else 'Representation Learning Model'} parameters: {agent_representation_model_total_params}\
        \n{'*'*20}")

    agent.set_mode("eval")

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, history_size=stateStackSize, seed=args['seed'], logger=logger)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, 4, 84, 84])

    steps = 0
    rall = 0
    rd = False
    intrinsic_reward_list = []
    while not rd:
        steps += 1
        actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            rall += r
            next_states = s.reshape([1, 4, 84, 84])
            next_obs = s[3, :, :].reshape([1, 1, 84, 84])

        # total reward = int reward + ext Reward
        intrinsic_reward = agent.compute_intrinsic_reward(next_obs)
        intrinsic_reward_list.append(intrinsic_reward)
        states = next_states[:, :, :, :]

        if rd:
            intrinsic_reward_list = (intrinsic_reward_list - np.mean(intrinsic_reward_list)) / np.std(
                intrinsic_reward_list)
            with open('int_reward', 'wb') as f:
                pickle.dump(intrinsic_reward_list, f)
            steps = 0
            rall = 0
