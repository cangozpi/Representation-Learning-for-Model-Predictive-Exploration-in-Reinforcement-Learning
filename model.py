import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from torch.nn import init
from config import default_config


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet"""

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise)

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(
            self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CnnActorCriticNetwork(nn.Module):
    # Refer to: https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py for the architecture

    # self.extracted_feature_embedding_dim  = 448
    extracted_feature_embedding_dim  = int(default_config['extracted_feature_embedding_dim'])

    def __init__(self, input_size, output_size, use_noisy_net=False):
        super(CnnActorCriticNetwork, self).__init__()

        if use_noisy_net:
            print('use NoisyNet')
            linear = NoisyLinear
        else:
            linear = nn.Linear

        # --------------- original paper's architecture below:
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4, # TODO: this equals StateStackSize
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.ReLU(),
            Flatten(),
            linear(
                7 * 7 * 64,
                256),
            nn.ReLU(),
            linear(
                256,
                CnnActorCriticNetwork.extracted_feature_embedding_dim), # = 448 # TODO: set extracted_feature_embedding_dim to 448 to match original RND paper !
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, CnnActorCriticNetwork.extracted_feature_embedding_dim),
            nn.ReLU(),
            linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, output_size)
        )

        self.extra_layer = nn.Sequential(
            linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, CnnActorCriticNetwork.extracted_feature_embedding_dim),
            nn.ReLU()
        )

        self.critic_ext = linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, 1)
        self.critic_int = linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, 1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        init.orthogonal_(self.critic_ext.weight, 0.01)
        self.critic_ext.bias.data.zero_()

        init.orthogonal_(self.critic_int.weight, 0.01)
        self.critic_int.bias.data.zero_()

        for i in range(len(self.actor)):
            if type(self.actor[i]) == nn.Linear:
                init.orthogonal_(self.actor[i].weight, 0.01)
                self.actor[i].bias.data.zero_()

        for i in range(len(self.extra_layer)):
            if type(self.extra_layer[i]) == nn.Linear:
                init.orthogonal_(self.extra_layer[i].weight, 0.1)
                self.extra_layer[i].bias.data.zero_()

        

        # --------------- my architecture below:
        # self.feature = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=4, # TODO: this equals StateStackSize
        #         out_channels=8,
        #         kernel_size=7,
        #         stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         in_channels=8,
        #         out_channels=8,
        #         kernel_size=5,
        #         stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(
        #         in_channels=8,
        #         out_channels=8,
        #         kernel_size=3,
        #         stride=1),
        #     nn.ReLU(),
        #     Flatten(),
        #     # -- old smaller layers below:
        #     # linear(
        #     #     # 7 * 7 * 64, #TOOD: ?
        #     #     # 23104 // 2,
        #     #     # 200,
        #     #     2048,
        #     #     64),
        #     # nn.ReLU(),
        #     # linear(
        #     #     64,
        #     #     32),
        #     # ---
        #     linear(
        #         # 7 * 7 * 64, #TOOD: ?
        #         # 23104 // 2,
        #         # 200,
        #         2048,
        #         256),
        #     nn.ReLU(),
        #     linear(
        #         256,
        #         CnnActorCriticNetwork.extracted_feature_embedding_dim), # = 128
        #     nn.ReLU()
        # )

        # self.actor = nn.Sequential(
        #     linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, CnnActorCriticNetwork.extracted_feature_embedding_dim),
        #     nn.ReLU(),
        #     linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, output_size)
        # )

        # self.extra_layer = nn.Sequential(
        #     linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, CnnActorCriticNetwork.extracted_feature_embedding_dim),
        #     nn.ReLU()
        # )

        # self.critic_ext = linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, 1)
        # self.critic_int = linear(CnnActorCriticNetwork.extracted_feature_embedding_dim, 1)
        

    def forward(self, state):
        x = self.feature(state)
        policy = self.actor(x)
        value_ext = self.critic_ext(self.extra_layer(x) + x)
        value_int = self.critic_int(self.extra_layer(x) + x)
        return policy, value_ext, value_int


class RNDModel(nn.Module):
    # Refer to: https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py for the architecture
    def __init__(self, input_size=32, output_size=512, train_method="modified_RND"):
        super(RNDModel, self).__init__()
        assert train_method in ['original_RND', 'modified_RND']

        self.input_size = input_size
        self.output_size = output_size

        if train_method == 'original_RND':
            feature_output = 7 * 7 * 64
            self.predictor = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                Flatten(),
                nn.Linear(feature_output, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size)
            )

            self.target = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                Flatten(),
                nn.Linear(feature_output, output_size)
            )

        elif train_method == 'modified_RND':
            self.predictor = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),

                nn.Linear(256, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, output_size)
            )

            self.target = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),

                nn.Linear(256, output_size),
            )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature
