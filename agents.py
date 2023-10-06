import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, RNDModel
from utils import global_grad_norm_
from utils import Logger
from config import default_config



class RNDAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            GAE_Lambda=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            max_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            representation_lr_method="BYOL",
            logger:Logger=None):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.GAE_Lambda = GAE_Lambda
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.max_grad_norm = max_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        assert isinstance(logger, Logger)
        self.logger = logger

        self.rnd = RNDModel(input_size, output_size)
        
        assert representation_lr_method in ['None', "BYOL", "Barlow-Twins"]
        self.representation_lr_method = representation_lr_method
        self.representation_model = None
        self.representation_loss_coef = 0
        # --------------------------------------------------------------------------------
        # for BYOL (Bootstrap Your Own Latent)
        if self.representation_lr_method == "BYOL":
            backbone_model = self.model.feature
            from BYOL import BYOL, Augment
            BYOL_projection_hidden_size = int(default_config['BYOL_projectionHiddenSize'])
            BYOL_projection_size = int(default_config['BYOL_projectionSize'])
            BYOL_moving_average_decay = float(default_config['BYOL_movingAverageDecay'])
            apply_same_transform_to_batch = default_config.getboolean('apply_same_transform_to_batch')
            self.representation_model = BYOL(backbone_model, in_features=448, projection_size=BYOL_projection_size, projection_hidden_size=BYOL_projection_hidden_size, moving_average_decay=BYOL_moving_average_decay, batch_norm_mlp=True, use_cuda=use_cuda) # Model used to perform representation learning (e.g. BYOL)
            self.data_transform = Augment(input_size, apply_same_transform_to_batch=apply_same_transform_to_batch)
            self.representation_loss_coef = float(default_config['BYOL_representationLossCoef'])
        # --------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        # for Barlow-Twins
        if self.representation_lr_method == "Barlow-Twins":
            backbone_model = self.model.feature
            from BarlowTwins import BarlowTwins, Transform
            import json
            projection_sizes = json.loads(default_config['BarlowTwinsProjectionSizes'])
            BarlowTwinsLambda = float(default_config['BarlowTwinsLambda'])
            apply_same_transform_to_batch = default_config.getboolean('apply_same_transform_to_batch')
            self.representation_model = BarlowTwins(backbone_model, in_features=448, projection_sizes=projection_sizes, lambd=BarlowTwinsLambda, use_cuda=use_cuda) # Model used to perform representation learning (e.g. BYOL)
            self.data_transform = Transform(input_size, apply_same_transform_to_batch=apply_same_transform_to_batch)
            self.representation_loss_coef = float(default_config['BarlowTwins_representationLossCoef'])
        # --------------------------------------------------------------------------------

        self.optimizer = optim.Adam(self.get_agent_parameters(), lr=learning_rate)

        self.rnd = self.rnd.to(self.device)
        self.model = self.model.to(self.device)
        if self.representation_model is not None:
            self.representation_model = self.representation_model.to(self.device)
    
    def get_agent_parameters(self):
        """
        Gathers the parameters (nn.Module parameters) of the RNDAgent and returns the unique ones 
        (without repetation of the shared parameters btw different models/modules).
        In other words returns parameters from PPO Agent, RND, and Representation Model (i.e. BYOL, Barlow-Twins).
        """
        if self.representation_model is not None:
            agent_params =  set(list(self.model.parameters()) + list(self.rnd.predictor.parameters()) + list(self.representation_model.get_trainable_parameters()))
        else:
            agent_params = set(set(list(self.model.parameters()) + list(self.rnd.predictor.parameters())))
        
        # double checking
        for p in self.model.parameters():
            assert p in agent_params
        for p in self.rnd.predictor.parameters():
            assert p in agent_params
        if self.representation_model is not None:
            for p in self.representation_model.get_trainable_parameters():
                assert p in agent_params

        return agent_params


    def set_mode(self, mode="train"):
        """
        Sets torch Modules (models) of the agent to the specified mode.
        """
        assert mode in ["train", "eval"]
        if mode == "train":
            self.model = self.model.train()
            self.rnd = self.rnd.train()
            if self.representation_model is not None:
                self.representation_model = self.representation_model.train()
        elif mode == "eval":
            self.model = self.model.eval()
            self.rnd = self.rnd.eval()
            if self.representation_model is not None:
                self.representation_model = self.representation_model.eval()
        else:
            raise NotImplementedError()

    def get_action(self, state):
        state = torch.Tensor(state).to(self.device)
        state = state.float()
        policy, value_ext, value_int = self.model(state)
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        action = self.random_choice_prob_index(action_prob)

        return action, value_ext.data.cpu().numpy().squeeze(), value_int.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def random_choice_prob_index(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def compute_intrinsic_reward(self, next_obs):
        next_obs = torch.FloatTensor(next_obs).to(self.device)

        target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)
        intrinsic_reward = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2

        return intrinsic_reward.data.cpu().numpy()

    def train_model(self, states, target_ext, target_int, y, adv, next_obs, old_policy, global_update):
        sample_range = np.arange(len(states))
        forward_mse = nn.MSELoss(reduction='none')

            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)

            total_loss, total_actor_loss, total_critic_loss, total_entropy_loss, total_rnd_loss, total_representation_loss = [], [], [], [], [], []
            total_grad_norm_unclipped = []
            if default_config['UseGradClipping']:
                total_grad_norm_clipped = []

            for j in range(int(len(states) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # Perform batching and sending to GPU:
                s_batch = torch.FloatTensor(states)[sample_idx].to(self.device)
                target_ext_batch = torch.FloatTensor(target_ext)[sample_idx].to(self.device)
                target_int_batch = torch.FloatTensor(target_int)[sample_idx].to(self.device)
                y_batch = torch.LongTensor(y)[sample_idx].to(self.device)
                adv_batch = torch.FloatTensor(adv)[sample_idx].to(self.device)
                next_obs_batch = torch.FloatTensor(next_obs)[sample_idx].to(self.device)
                with torch.no_grad():
                    policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size)[sample_idx].to(self.device)
                    m_old = Categorical(F.softmax(policy_old_list, dim=-1))
                    log_prob_old = m_old.log_prob(y_batch)


                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch)

                rnd_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(rnd_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                rnd_loss = (rnd_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # ---------------------------------------------------------------------------------


                representation_loss = 0
                # --------------------------------------------------------------------------------
                # for BYOL (Bootstrap Your Own Latent):
                if self.representation_lr_method == "BYOL":
                    # sample image transformations and transform the images to obtain the 2 views
                    B, STATE_STACK_SIZE, H, W = s_batch.shape
                    if default_config.getboolean('apply_same_transform_to_batch'):
                        s_batch_views = self.data_transform(torch.reshape(s_batch, [-1, H, W])[:, None, :, :]) # -> [B*STATE_STACK_SIZE, C=1, H, W], [B*STATE_STACK_SIZE, C=1, H, W]
                    else:
                        s_batch_views = self.data_transform(s_batch) # -> [B*STATE_STACK_SIZE, C=STATE_STACK_SIZE, H, W], [B, C=STATE_STACK_SIZE, H, W]
                    s_batch_view1, s_batch_view2 = torch.reshape(s_batch_views[0], [B, STATE_STACK_SIZE, H, W]), \
                        torch.reshape(s_batch_views[1], [B, STATE_STACK_SIZE, H, W]) # -> [B, STATE_STACK_SIZE, H, W], [B, STATE_STACK_SIZE, H, W]
                
                    assert self.representation_model.net is self.model.feature # make sure that BYOL net and RL algo's feature extractor both point to the same network

                    # plot original frame vs transformed views for debugging purposes
                    if False:
                        import matplotlib.pyplot as plt
                        for i in range(4):
                            idx = np.random.choice(B)
                            print(idx)
                            fig, axs = plt.subplots(4, 2, constrained_layout=True)
                            axs[0,0].imshow(s_batch[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[0,1].imshow(s_batch_view1[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,0].imshow(s_batch[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,1].imshow(s_batch_view1[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,0].imshow(s_batch[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,1].imshow(s_batch_view1[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,0].imshow(s_batch[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,1].imshow(s_batch_view1[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')

                            axs[0,0].set_title(f'original state:{idx} frame:0', fontsize=10)
                            axs[0,1].set_title(f'augmented state:{idx} frame:0', fontsize=10)
                            axs[1,0].set_title(f'original state:{idx} frame:1', fontsize=10)
                            axs[1,1].set_title(f'augmented state:{idx} frame:1', fontsize=10)
                            axs[2,0].set_title(f'original state:{idx} frame:2', fontsize=10)
                            axs[2,1].set_title(f'augmented state:{idx} frame:2', fontsize=10)
                            axs[3,0].set_title(f'original state:{idx} frame:3', fontsize=10)
                            axs[3,1].set_title(f'augmented state:{idx} frame:3', fontsize=10)
                            plt.show()

                    # compute BYOL loss
                    BYOL_loss = self.representation_model(s_batch_view1, s_batch_view2) 
                    representation_loss = BYOL_loss
                # ---------------------------------------------------------------------------------


                # --------------------------------------------------------------------------------
                # for Barlow-Twins:
                if self.representation_lr_method == "Barlow-Twins":
                    # sample image transformations and transform the images to obtain the 2 views
                    B, STATE_STACK_SIZE, H, W = s_batch.shape
                    if default_config.getboolean('apply_same_transform_to_batch'):
                        s_batch_views = self.data_transform(torch.reshape(s_batch, [-1, H, W])[:, None, :, :]) # -> [B*STATE_STACK_SIZE, C=1, H, W], [B*STATE_STACK_SIZE, C=1, H, W]
                    else:
                        s_batch_views = self.data_transform(s_batch) # -> [B, C=STATE_STACK_SIZE, H, W], [B, C=STATE_STACK_SIZE, H, W]
                    s_batch_view1, s_batch_view2 = torch.reshape(s_batch_views[0], [B, STATE_STACK_SIZE, H, W]), \
                        torch.reshape(s_batch_views[1], [B, STATE_STACK_SIZE, H, W]) # -> [B, STATE_STACK_SIZE, H, W], [B, STATE_STACK_SIZE, H, W]
                
                    assert self.representation_model.backbone is self.model.feature # make sure that Barlow-Twins backbone and RL algo's feature extractor both point to the same network

                    # plot original frame vs transformed views for debugging purposes
                    if False:
                        import matplotlib.pyplot as plt
                        for i in range(4):
                            idx = np.random.choice(B)
                            print(idx)
                            fig, axs = plt.subplots(4, 2, constrained_layout=True)
                            axs[0,0].imshow(s_batch[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[0,1].imshow(s_batch_view1[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,0].imshow(s_batch[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,1].imshow(s_batch_view1[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,0].imshow(s_batch[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,1].imshow(s_batch_view1[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,0].imshow(s_batch[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,1].imshow(s_batch_view1[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')

                            axs[0,0].set_title(f'original state:{idx} frame:0', fontsize=10)
                            axs[0,1].set_title(f'augmented state:{idx} frame:0', fontsize=10)
                            axs[1,0].set_title(f'original state:{idx} frame:1', fontsize=10)
                            axs[1,1].set_title(f'augmented state:{idx} frame:1', fontsize=10)
                            axs[2,0].set_title(f'original state:{idx} frame:2', fontsize=10)
                            axs[2,1].set_title(f'augmented state:{idx} frame:2', fontsize=10)
                            axs[3,0].set_title(f'original state:{idx} frame:3', fontsize=10)
                            axs[3,1].set_title(f'augmented state:{idx} frame:3', fontsize=10)
                            plt.show()

                    # compute Barlow-Twins loss
                    BarlowTwins_loss = self.representation_model(s_batch_view1, s_batch_view2) 
                    representation_loss = BarlowTwins_loss
                # ---------------------------------------------------------------------------------


                # --------------------------------------------------------------------------------
                # for Proximal Policy Optimization (PPO):
                policy, value_ext, value_int = self.model(s_batch)
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch)

                ratio = torch.exp(log_prob - log_prob_old)

                surr1 = ratio * adv_batch
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch)
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch)

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean()
                # --------------------------------------------------------------------------------

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + rnd_loss + self.representation_loss_coef * representation_loss
                loss.backward()

                grad_norm_unclipped = global_grad_norm_(self.get_agent_parameters())
                if default_config['UseGradClipping']:
                    nn.utils.clip_grad_norm_(self.get_agent_parameters(), self.max_grad_norm) # gradient clipping
                    grad_norm_clipped = global_grad_norm_(self.get_agent_parameters())
                # Log final model grads in detail
                if i == self.epoch - 1:
                    self.logger.log_gradients_in_model_to_tb(self.model, global_update + i, f'PPO', log_full_detail=True)
                    self.logger.log_gradients_in_model_to_tb(self.rnd, global_update + i, f'RND', log_full_detail=True)
                    if self.representation_model is not None:
                        self.logger.log_gradients_in_model_to_tb(self.representation_model, global_update + i, f'{self.representation_lr_method}', log_full_detail=True)

                self.optimizer.step()

                # logging
                total_loss.append(loss.detach().cpu().item())
                total_actor_loss.append(actor_loss.detach().cpu().item())
                total_critic_loss.append(0.5 * critic_loss.detach().cpu().item())
                total_entropy_loss.append(- self.ent_coef * entropy.detach().cpu().item())
                total_rnd_loss.append(rnd_loss.detach().cpu().item())
                if self.representation_model is not None:
                    total_representation_loss.append(self.representation_loss_coef * representation_loss.detach().cpu().item())
                total_grad_norm_unclipped.append(grad_norm_unclipped)
                if default_config['UseGradClipping']:
                    total_grad_norm_clipped.append(grad_norm_clipped)

                # EMA update BYOL target network params
                if self.representation_lr_method == "BYOL":
                    self.representation_model.update_moving_average()
            
            # logging
            if self.logger is not None:
                self.logger.log_scalar_to_tb_with_step('train/overall_loss (everything combined) vs parameter_update', np.mean(total_loss), global_update + i)
                self.logger.log_scalar_to_tb_with_step('train/PPO_actor_loss vs parameter_update', np.mean(total_actor_loss), global_update + i)
                self.logger.log_scalar_to_tb_with_step('train/PPO_critic_loss vs parameter_update', np.mean(total_critic_loss), global_update + i)
                self.logger.log_scalar_to_tb_with_step('train/PPO_entropy_loss vs parameter_update', np.mean(total_entropy_loss), global_update + i)
                self.logger.log_scalar_to_tb_with_step('train/RND_loss vs parameter_update', np.mean(total_rnd_loss), global_update + i)
                if self.representation_model is not None:
                    self.logger.log_scalar_to_tb_with_step(f'train/Representation_loss({self.representation_lr_method}) vs parameter_update', np.mean(total_representation_loss), global_update + i)
                self.logger.log_scalar_to_tb_with_step('grads/grad_norm_unclipped', np.mean(total_grad_norm_unclipped), global_update + i)
                if default_config['UseGradClipping']:
                    self.logger.log_scalar_to_tb_with_step('grads/grad_norm_clipped', np.mean(total_grad_norm_clipped), global_update + i)
                # Log final model parameters in detail
                self.logger.log_parameters_in_model_to_tb(self.model, global_update + i, f'PPO')
                self.logger.log_parameters_in_model_to_tb(self.rnd, global_update + i, f'RND')
                if self.representation_model is not None:
                    self.logger.log_parameters_in_model_to_tb(self.representation_model, global_update + i, f'{self.representation_lr_method}')
