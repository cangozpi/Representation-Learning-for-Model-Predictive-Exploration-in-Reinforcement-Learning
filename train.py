from agents import *
from envs import *
from utils import *
from config import *
from torch.multiprocessing import Pipe

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import numpy as np


def main():
    print({section: dict(config[section]) for section in config.sections()})
    train_method = default_config['TrainMethod']
    representation_lr_method = str(default_config['representationLearningMethod'])

    env_id = default_config['EnvID']
    env_type = default_config['EnvType']

    if env_type == 'mario':
        env = BinarySpaceToDiscreteSpaceEnv(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    elif env_type == 'atari':
        env = gym.make(env_id)
    else:
        raise NotImplementedError
    if default_config['PreProcHeight'] is not None:
        input_size = int(default_config['PreProcHeight'])
    else:
        input_size = env.observation_space.shape  # 4
    output_size = env.action_space.n  # 2

    if 'Breakout' in env_id:
        output_size -= 1

    env.close()

    is_load_model = default_config.getboolean('loadModel')
    is_render = default_config.getboolean('render')
    model_path = 'models/{}.model'.format(env_id)
    predictor_path = 'models/{}.pred'.format(env_id)
    target_path = 'models/{}.target'.format(env_id)
    BYOL_model_path = 'models/{}.BYOLModelPath'.format(env_id)
    BarlowTwins_model_path = 'models/{}.BarlowTwinsModelPath'.format(env_id)

    writer = SummaryWriter()

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
    clip_grad_norm = float(default_config['ClipGradNorm'])
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
        logger=writer
    )


    if is_load_model:
        print('load model...')
        if use_cuda:
            agent.model.load_state_dict(torch.load(model_path))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path))
            agent.rnd.target.load_state_dict(torch.load(target_path))
            if representation_lr_method == "BYOL": # BYOL
                agent.representation_model.load_state_dict(torch.load(BYOL_model_path))
                agent.representation_model.net = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
            if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                agent.representation_model.load_state_dict(torch.load(BarlowTwins_model_path))
                agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo

        else:
            agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            agent.rnd.predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
            agent.rnd.target.load_state_dict(torch.load(target_path, map_location='cpu'))
            if representation_lr_method == "BYOL": # BYOL
                agent.representation_model.load_state_dict(torch.load(BYOL_model_path, map_location='cpu'))
                agent.representation_model.net = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
            if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                agent.representation_model.load_state_dict(torch.load(BarlowTwins_model_path, map_location='cpu'))
                agent.representation_model.backbone = agent.model.feature # representation_model's net should map to the feature extractor of the RL algo
        print('load finished!')

    
    agent_PPO_total_params = sum(p.numel() for p in agent.model.parameters())
    agent_RND_predictor_total_params = sum(p.numel() for p in agent.rnd.predictor.parameters())
    agent_representation_model_total_params = sum(p.numel() for p in agent.representation_model.parameters()) if agent.representation_model is not None else 0
    print(f"{'*'*20}\
        \nNumber of PPO parameters: {agent_PPO_total_params}\
        \nNumber of RND_predictor parameters: {agent_RND_predictor_total_params}\
        \nNumber of {representation_lr_method if representation_lr_method != 'None' else 'Representation Learning Model'} parameters: {agent_representation_model_total_params}\
        \n{'*'*20}")


    works = []
    parent_conns = []
    child_conns = []
    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        work = env_type(env_id, is_render, idx, child_conn, sticky_action=sticky_action, p=action_prob,
                        life_done=life_done, history_size=stateStackSize)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    states = np.zeros([num_worker, stateStackSize, input_size, input_size])

    sample_episode = 0
    sample_rall = 0
    sample_step = 0
    sample_env_idx = 0
    sample_i_rall = 0
    global_update = 0
    global_step = 0

    # normalize obs
    print('Start to initialize observation normalization parameter.....')
    next_obs = []
    for step in range(num_step * pre_obs_norm_step):
        actions = np.random.randint(0, output_size, size=(num_worker,))

        for parent_conn, action in zip(parent_conns, actions):
            parent_conn.send(action)

        for parent_conn in parent_conns:
            s, r, d, rd, lr = parent_conn.recv()
            next_obs.append(s[stateStackSize - 1, :, :].reshape([1, input_size, input_size]))

        if len(next_obs) % (num_step * num_worker) == 0:
            next_obs = np.stack(next_obs)
            obs_rms.update(next_obs)
            next_obs = []
    print('End to initialize...')

    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_int_reward, total_next_obs, total_ext_values, total_int_values, total_policy, total_policy_np = \
            [], [], [], [], [], [], [], [], [], [], []
        global_step += (num_worker * num_step)
        global_update += 1

        # Step 1. n-step rollout
        for _ in range(num_step):
            actions, value_ext, value_int, policy = agent.get_action(np.float32(states) / 255.)

            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            next_states, rewards, dones, real_dones, log_rewards, next_obs = [], [], [], [], [], []
            for parent_conn in parent_conns:
                s, r, d, rd, lr = parent_conn.recv()
                next_states.append(s)
                rewards.append(r)
                dones.append(d)
                real_dones.append(rd) # --> [num_env]
                log_rewards.append(lr) # --> [num_env]
                next_obs.append(s[(stateStackSize - 1), :, :].reshape([1, input_size, input_size]))

            next_states = np.stack(next_states) # -> [num_env, state_stack_size, H, W]
            rewards = np.hstack(rewards) # -> [num_env, ]
            dones = np.hstack(dones) # -> [num_env, ]
            real_dones = np.hstack(real_dones) # -> [num_env, ]
            next_obs = np.stack(next_obs) # -> [num_env, 1, H, W]

            # total reward = int reward + ext Reward
            intrinsic_reward = agent.compute_intrinsic_reward(
                ((next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5)) # -> [num_env, ]
            intrinsic_reward = np.hstack(intrinsic_reward)
            sample_i_rall += intrinsic_reward[sample_env_idx]

            total_next_obs.append(next_obs) # --> [num_step, num_env, state_stack_size, H, W]
            total_int_reward.append(intrinsic_reward)
            total_state.append(states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_ext_values.append(value_ext)
            total_int_values.append(value_int)
            total_policy.append(policy)
            total_policy_np.append(policy.cpu().numpy())

            states = next_states[:, :, :, :]

            sample_rall += log_rewards[sample_env_idx]

            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                writer.add_scalar('data/reward_per_epi', sample_rall, sample_episode)
                writer.add_scalar('data/reward_per_rollout', sample_rall, global_update)
                writer.add_scalar('data/step', sample_step, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_i_rall = 0

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
        total_logging_policy = np.vstack(total_policy_np) # --> [num_env * num_step, output_size]


        # writer.add_scalar('data/mean_episodic_return (extrinsic) vs episode', np.sum(total_reward) / num_worker, sample_episode)
        writer.add_scalar('data/mean_episodic_return (extrinsic) vs parameter_update', np.sum(total_reward) / num_worker, global_update)

        # Step 2. calculate intrinsic reward
        # running mean intrinsic reward
        total_int_reward = np.stack(total_int_reward).transpose() # --> [num_env, num_step]
        total_reward_per_env = np.array([discounted_reward.update(reward_per_step) for reward_per_step in
                                         total_int_reward.T])
        mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
        reward_rms.update_from_moments(mean, std ** 2, count) # update reward normalization parameters using intrinsic rewards

        # normalize intrinsic reward
        total_int_reward /= np.sqrt(reward_rms.var)
        # writer.add_scalar('data/mean_episodic_return (intrinsic) vs episode', np.sum(total_int_reward) / num_worker, sample_episode)
        writer.add_scalar('data/mean_episodic_return (intrinsic) vs parameter_update', np.sum(total_int_reward) / num_worker, global_update)
        # writer.add_scalar('data/int_reward_per_epi', np.sum(total_int_reward) / num_worker, sample_episode)
        # writer.add_scalar('data/int_reward_per_rollout', np.sum(total_int_reward) / num_worker, global_update)
        # -------------------------------------------------------------------------------------------

        # logging Max action probability
        writer.add_scalar('data/max_policy_action_prob vs episode', softmax(total_logging_policy).max(1).mean(), sample_episode)

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
        print("YOOO ENTERED TRAINIG:JJ")
        agent.train_model(np.float32(total_state) / 255., ext_target, int_target, total_action,
                          total_adv, ((total_next_obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5),
                          total_policy, global_update)
        print("YOOO EXITTED TRAINIG:JJ")

        # if global_step % (num_worker * num_step * 100) == 0:
        if True:
            print('Now Global Step :{}'.format(global_step))
            torch.save(agent.model.state_dict(), model_path)
            torch.save(agent.rnd.predictor.state_dict(), predictor_path)
            torch.save(agent.rnd.target.state_dict(), target_path)
            if representation_lr_method == "BYOL": # BYOL
                torch.save(agent.representation_model.state_dict(), BYOL_model_path)
            if representation_lr_method == "Barlow-Twins": # Barlow-Twins
                torch.save(agent.representation_model.state_dict(), BarlowTwins_model_path)


if __name__ == '__main__':
    main()
