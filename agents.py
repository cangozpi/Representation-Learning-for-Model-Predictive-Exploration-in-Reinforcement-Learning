import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim

from torch.distributions.categorical import Categorical

from model import CnnActorCriticNetwork, RNDModel
from utils import global_grad_norm_



class RNDAgent(object):
    def __init__(
            self,
            input_size,
            output_size,
            num_env,
            num_step,
            gamma,
            lam=0.95,
            learning_rate=1e-4,
            ent_coef=0.01,
            clip_grad_norm=0.5,
            epoch=3,
            batch_size=128,
            ppo_eps=0.1,
            update_proportion=0.25,
            use_gae=True,
            use_cuda=False,
            use_noisy_net=False,
            representation_lr_method="BYOL"):
        self.model = CnnActorCriticNetwork(input_size, output_size, use_noisy_net)
        self.num_env = num_env
        self.output_size = output_size
        self.input_size = input_size
        self.num_step = num_step
        self.gamma = gamma
        self.lam = lam
        self.epoch = epoch
        self.batch_size = batch_size
        self.use_gae = use_gae
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.clip_grad_norm = clip_grad_norm
        self.update_proportion = update_proportion
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.rnd = RNDModel(input_size, output_size)
        
        assert representation_lr_method in [None, "BYOL", "Barlow-Twins"]
        self.representation_lr_method = representation_lr_method
        # --------------------------------------------------------------------------------
        # for BYOL (Bootstrap Your Own Latent)
        if self.representation_lr_method == "BYOL":
            backbone_model = self.model.feature
            from BYOL import BYOL, Augment
            self.representation_model = BYOL(backbone_model, in_features=448, batch_norm_mlp=True, use_cuda=use_cuda) # Model used to perform representation learning (e.g. BYOL)
            self.data_transform = Augment(input_size)
        # --------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------
        # for Barlow-Twins
        if self.representation_lr_method == "Barlow-Twins":
            backbone_model = self.model.feature
            from BarlowTwins import BarlowTwins, Transform
            projection_sizes = [512, 512, 512, 512]
            lambd = 0.5
            self.representation_model = BarlowTwins(backbone_model, in_features=448, projection_sizes=projection_sizes, lambd=lambd, use_cuda=use_cuda) # Model used to perform representation learning (e.g. BYOL)
            self.data_transform = Transform(input_size)
        # --------------------------------------------------------------------------------

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.rnd.predictor.parameters()) + list(self.representation_model.parameters()),
                                    lr=learning_rate)

        self.rnd = self.rnd.to(self.device)
        self.model = self.model.to(self.device)
        self.representation_model = self.representation_model.to(self.device)


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

    def train_model(self, s_batch, target_ext_batch, target_int_batch, y_batch, adv_batch, next_obs_batch, old_policy):
        s_batch = torch.FloatTensor(s_batch).to(self.device)
        target_ext_batch = torch.FloatTensor(target_ext_batch).to(self.device)
        target_int_batch = torch.FloatTensor(target_int_batch).to(self.device)
        y_batch = torch.LongTensor(y_batch).to(self.device)
        adv_batch = torch.FloatTensor(adv_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)

        sample_range = np.arange(len(s_batch))
        forward_mse = nn.MSELoss(reduction='none')

        with torch.no_grad():
            policy_old_list = torch.stack(old_policy).permute(1, 0, 2).contiguous().view(-1, self.output_size).to(
                self.device)

            m_old = Categorical(F.softmax(policy_old_list, dim=-1))
            log_prob_old = m_old.log_prob(y_batch)
            # ------------------------------------------------------------

        for i in range(self.epoch):
            np.random.shuffle(sample_range)
            for j in range(int(len(s_batch) / self.batch_size)):
                sample_idx = sample_range[self.batch_size * j:self.batch_size * (j + 1)]

                # --------------------------------------------------------------------------------
                # for Curiosity-driven(Random Network Distillation)
                predict_next_state_feature, target_next_state_feature = self.rnd(next_obs_batch[sample_idx])

                forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(-1)
                # Proportion of exp used for predictor update
                mask = torch.rand(len(forward_loss)).to(self.device)
                mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
                forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
                # ---------------------------------------------------------------------------------


                representation_loss = 0
                # --------------------------------------------------------------------------------
                # for BYOL (Bootstrap Your Own Latent):
                if self.representation_lr_method == "BYOL":
                    # sample image transformations and transform the images to obtain the 2 views
                    B, STATE_STACK_SIZE, H, W = s_batch.shape
                    s_batch_views = self.data_transform(torch.reshape(s_batch, [-1, H, W])[:, None, :, :]) # -> [B*STATE_STACK_SIZE, C=1, H, W], [B*STATE_STACK_SIZE, C=1, H, W]
                    s_batch_view1, s_batch_view2 = torch.reshape(s_batch_views[0], [B, STATE_STACK_SIZE, H, W]), \
                        torch.reshape(s_batch_views[1], [B, STATE_STACK_SIZE, H, W]) # -> [B, STATE_STACK_SIZE, H, W], [B, STATE_STACK_SIZE, H, W]
                
                    assert self.representation_model.net is self.model.feature # make sure that BYOL net and RL algo's feature extractor both point to the same network

                    # plot original frame vs transformed views for debugging purposes
                    if False:
                        import matplotlib.pyplot as plt
                        for i in range(4):
                            idx = np.random.choice(B)
                            print(idx)
                            fig, axs = plt.subplots(4, 2)
                            axs[0,0].imshow(s_batch[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[0,1].imshow(s_batch_view1[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,0].imshow(s_batch[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,1].imshow(s_batch_view1[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,0].imshow(s_batch[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,1].imshow(s_batch_view1[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,0].imshow(s_batch[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,1].imshow(s_batch_view1[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')
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
                    s_batch_views = self.data_transform(torch.reshape(s_batch, [-1, H, W])[:, None, :, :]) # -> [B*STATE_STACK_SIZE, C=1, H, W], [B*STATE_STACK_SIZE, C=1, H, W]
                    s_batch_view1, s_batch_view2 = torch.reshape(s_batch_views[0], [B, STATE_STACK_SIZE, H, W]), \
                        torch.reshape(s_batch_views[1], [B, STATE_STACK_SIZE, H, W]) # -> [B, STATE_STACK_SIZE, H, W], [B, STATE_STACK_SIZE, H, W]
                
                    assert self.representation_model.backbone is self.model.feature # make sure that Barlow-Twins backbone and RL algo's feature extractor both point to the same network

                    # plot original frame vs transformed views for debugging purposes
                    if False:
                        import matplotlib.pyplot as plt
                        for i in range(4):
                            idx = np.random.choice(B)
                            print(idx)
                            fig, axs = plt.subplots(4, 2)
                            axs[0,0].imshow(s_batch[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[0,1].imshow(s_batch_view1[idx, 0, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,0].imshow(s_batch[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[1,1].imshow(s_batch_view1[idx, 1, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,0].imshow(s_batch[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[2,1].imshow(s_batch_view1[idx, 2, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,0].imshow(s_batch[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')
                            axs[3,1].imshow(s_batch_view1[idx, 3, None, :, :].permute(1, 2, 0), cmap='gray')
                            plt.show()

                    # compute Barlow-Twins loss
                    BarlowTwins_loss = self.representation_model(s_batch_view1, s_batch_view2) 
                    representation_loss = BarlowTwins_loss
                # ---------------------------------------------------------------------------------


                # --------------------------------------------------------------------------------
                # for Proximal Policy Optimization(PPO):
                policy, value_ext, value_int = self.model(s_batch[sample_idx])
                m = Categorical(F.softmax(policy, dim=-1))
                log_prob = m.log_prob(y_batch[sample_idx])

                ratio = torch.exp(log_prob - log_prob_old[sample_idx])

                surr1 = ratio * adv_batch[sample_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.ppo_eps,
                    1.0 + self.ppo_eps) * adv_batch[sample_idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_ext_loss = F.mse_loss(value_ext.sum(1), target_ext_batch[sample_idx])
                critic_int_loss = F.mse_loss(value_int.sum(1), target_int_batch[sample_idx])

                critic_loss = critic_ext_loss + critic_int_loss

                entropy = m.entropy().mean()
                # --------------------------------------------------------------------------------

                self.optimizer.zero_grad()
                loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + forward_loss + representation_loss
                loss.backward()
                global_grad_norm_(list(self.model.parameters())+list(self.rnd.predictor.parameters()))
                self.optimizer.step()

                # EMA update BYOL target network params
                if self.representation_lr_method == "BYOL":
                    self.representation_model.update_moving_average()
