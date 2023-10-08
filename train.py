from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

import numpy as np
from utils import Logger, set_seed
from os import path
from collections import deque


def main(args):
    set_seed(args['seed']) # Note: this will not seed the gym environment

    logger = Logger(file_log_path=path.join("logs", "file_logs", args['log_name']), tb_log_path=path.join("logs", "tb_logs", args['log_name']))
    logger.log_msg_to_both_console_and_file(str(dict(**{section: dict(config[section]) for section in config.sections()}, **args)))

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

    use_cuda = default_config.getboolean('UseGPU')
    use_gae = default_config.getboolean('UseGAE')
    use_noisy_net = default_config.getboolean('UseNoisyNet')

    GAE_Lambda = float(default_config['GAELambda'])
    num_worker = int(default_config['NumEnv'])

    num_step = int(default_config['NumStep'])

    ppo_eps = float(default_config['PPOEps'])
    epoch = int(default_config['Epoch'])
    mini_batch = int(default_config['MiniBatch'])
    batch_size = int(num_step * num_worker / mini_batch)
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
    obs_rms = RunningMeanStd(shape=(1, 1, input_size, input_size)) # used for normalizing observations
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

    agent = agent(
        input_size,
        output_size,
        num_worker,
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
        logger=logger
    )

    global_update = 0
    global_step = 0
    undiscounted_episode_return = deque([], maxlen=100)
    episode_lengths = deque([], maxlen=100)
    if 'Montezuma' in env_id:
        visited_rooms = deque([], maxlen=100)

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
            agent.optimizer.load_state_dict(load_checkpoint['agent.optimizer.state_dict'])

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
            agent.optimizer.load_state_dict(load_checkpoint['agent.optimizer.state_dict'])

        obs_rms = load_checkpoint['obs_rms']
        reward_rms = load_checkpoint['reward_rms']
        global_update = load_checkpoint['global_update']
        global_step = load_checkpoint['global_step']
        undiscounted_episode_return = load_checkpoint['undiscounted_episode_return']
        episode_lengths = load_checkpoint['episode_lengths']
        if 'Montezuma' in env_id:
            visited_rooms = load_checkpoint['visited_rooms']
        logger.tb_global_steps = load_checkpoint['logger.tb_global_steps']

        logger.log_msg_to_both_console_and_file('loading finished!')

    
    agent_PPO_total_params = sum(p.numel() for p in agent.model.parameters())
    agent_RND_predictor_total_params = sum(p.numel() for p in agent.rnd.predictor.parameters())
    agent_representation_model_total_params = sum(p.numel() for p in agent.representation_model.parameters()) if agent.representation_model is not None else 0
    logger.log_msg_to_both_console_and_file(f"{'*'*20}\
        \nNumber of PPO parameters: {agent_PPO_total_params}\
        \nNumber of RND_predictor parameters: {agent_RND_predictor_total_params}\
        \nNumber of {representation_lr_method if representation_lr_method != 'None' else 'Representation Learning Model'} parameters: {agent_representation_model_total_params}\
        \n{'*'*20}")

    agent.set_mode("train")

    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, history_size=stateStackSize, seed=args['seed']+idx, logger=logger) # Note that seed+idx is required to make parallel envs play different scenarios
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, stateStackSize, input_size, input_size])


    # normalize obs
    if is_load_model == False:
        logger.log_msg_to_both_console_and_file('Start to initialize observation normalization parameter.....')
        next_obs = []
        for step in range(num_step * pre_obs_norm_step):
            actions = np.random.randint(0, output_size, size=(num_worker,))

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            for parent_conn in parent_conns:
                s, r, d, _, info = parent_conn.recv()
                next_obs.append(s[stateStackSize - 1, :, :].reshape([1, input_size, input_size]))

            if len(next_obs) % (num_step * num_worker) == 0:
                next_obs = np.stack(next_obs)
                obs_rms.update(next_obs)
                next_obs = []
        logger.log_msg_to_both_console_and_file('End to initialize...')

    while True:
        total_state, total_reward, total_done, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy = \
            [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, next_obs = [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, _, info, = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d) # --> [num_env]
                next_obs.append(s[(stateStackSize - 1), :, :].reshape([1, input_size, input_size]))
                if 'episode' in info:
                    if 'Montezuma' in env_id:
                        visited_rooms.append(info['episode']['visited_rooms'])
                    undiscounted_episode_return.append(info['episode']['undiscounted_episode_return'])
                    episode_lengths.append(info['episode']['l'])
                    

            next_states = np.stack(next_states) # -> [num_env, state_stack_size, H, W]
            rewards = np.hstack(rewards) # -> [num_env, ]
            dones = np.hstack(dones) # -> [num_env, ]
            next_obs = np.stack(next_obs) # -> [num_env, 1, H, W]

            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]
            intrinsic_reward = np.hstack(intrinsic_reward)

            total_next_obs.append(next_obs) # --> [num_step, num_env, state_stack_size, H, W]
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)

            states = next_states[:, :, :, :]


        # calculate last next value
        _, value_ext, value_int, _ = agent.get_action(np.float32(states) / 255.)
        total_ext_values.append(value_ext)
        total_int_values.append(value_int)
        # --------------------------------------------------

        total_state = np.stack(total_state).transpose([1, 0, 2, 3, 4]).reshape([-1, stateStackSize, input_size, input_size]) # --> [num_env * num_step, state_stack_size, H, W]
        total_reward = np.stack(total_reward).transpose().clip(-1, 1) # --> [num_env, num_step]
        total_action = np.stack(total_action).transpose().reshape([-1]) # --> [num_env * num_step]
        total_done = np.stack(total_done).transpose() # --> [num_env, num_step]
        total_next_obs = np.stack(total_next_obs).transpose([1, 0, 2, 3, 4]).reshape([-1, 1, input_size, input_size]) # --> [num_env * num_step, 1, H, W]
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
                                              num_worker)

        # intrinsic reward calculate
        # None Episodic
        int_target, int_adv = make_train_data(total_int_reward,
                                              np.zeros_like(total_int_reward),
                                              total_int_values,
                                              int_gamma,
                                              num_step,
                                              num_worker)

        # add ext adv and int adv
        total_adv = int_adv * int_coef + ext_adv * ext_coef
        # -----------------------------------------------

        # Step 4. update obs normalize param
        obs_rms.update(total_next_obs)
        # -----------------------------------------------

        # Step 5. Training!
        logger.log_msg_to_both_console_and_file("ENTERED TRAINING:")
        agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy, global_update)
        logger.log_msg_to_both_console_and_file("EXITTED TRAINING")


        # Logging
        logger.log_scalar_to_tb_with_step('data/Mean of rollout rewards (extrinsic) vs Parameter updates', np.mean(total_reward), global_update)
        logger.log_scalar_to_tb_with_step('data/Mean of rollout rewards (intrinsic) vs Parameter updates', np.mean(total_int_reward), global_update)
        if len(episode_lengths) > 0: # check if any episode has been completed yet
            logger.log_scalar_to_tb_with_step('data/Mean undiscounted episodic return (over last 100 episodes) (extrinsic) vs Parameter updates', np.mean(undiscounted_episode_return), global_update)
            logger.log_scalar_to_tb_with_step('data/Mean episode lengths (over last 100 episodes) vs Parameter updates', np.mean(episode_lengths), global_update)
            if 'Montezuma' in env_id:
                logger.log_scalar_to_tb_with_step('data/Mean number of rooms found (over last 100 episodes) vs Parameter updates', np.mean(list(map(lambda x: len(x),visited_rooms))), global_update)


        # Save checkpoint
        if global_step % (num_worker * num_step * int(default_config["saveCkptEvery"])) == 0:
            ckpt_dict = {
                **{
                    'agent.optimizer.state_dict': agent.optimizer.state_dict(),
                    'agent.model.state_dict': agent.model.state_dict(),
                    'agent.rnd.predictor.state_dict': agent.rnd.predictor.state_dict(),
                    'agent.rnd.target.state_dict': agent.rnd.target.state_dict(),
                },
                **({'agent.representation_model.state_dict': agent.representation_model.state_dict()} if representation_lr_method == "BYOL" else {}),
                **({'agent.representation_model.state_dict': agent.representation_model.state_dict()} if representation_lr_method == "Barlow-Twins" else {}),
                **{
                    'obs_rms': obs_rms,
                    'reward_rms': reward_rms,
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
                ckpt_dict.update(visited_rooms=visited_rooms)
            os.makedirs('/'.join(save_ckpt_path.split('/')[:-1]), exist_ok=True)
            torch.save(ckpt_dict, save_ckpt_path)
            logger.log_msg_to_both_console_and_file('Saved ckpt at Global Step :{}'.format(global_step))

