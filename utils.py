from config import *
import numpy as np

import torch
# from torch._six import inf
from torch import inf

import random
import logging
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import os
import datetime
from os import path
from matplotlib import pyplot as plt

# train_method = default_config['TrainMethod']
# if default_config['TrainMethod'] in ['PPO', 'ICM', 'RND']:
#     num_step = int(ppo_config['NumStep'])
# else:
#     num_step = int(default_config['NumStep'])

use_gae = default_config.getboolean('UseGAE')
GAE_lambda = float(default_config['GAELambda'])

def init_tb_global_step():
    return 0


def make_train_data(reward, done, value, gamma, num_step, num_worker):

    discounted_return = np.empty([num_worker, num_step])

    # Discounted Return
    if use_gae:
        gae = np.zeros_like([num_worker, ])
        for t in range(num_step - 1, -1, -1):
            delta = reward[:, t] + gamma * value[:, t + 1] * (1 - done[:, t]) - value[:, t]
            gae = delta + gamma * GAE_lambda * (1 - done[:, t]) * gae

            discounted_return[:, t] = gae + value[:, t]

            # For Actor
        adv = discounted_return - value[:, :-1]

    else:
        running_add = value[:, -1]
        for t in range(num_step - 1, -1, -1): # = [(num_step - 1), ..., 0]
            running_add = reward[:, t] + gamma * running_add * (1 - done[:, t])
            discounted_return[:, t] = running_add

        # For Actor
        adv = discounted_return - value[:, :-1]

    return discounted_return.reshape([-1]), adv.reshape([-1])


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def global_grad_norm_(parameters, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

    return total_norm


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`, gym.
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    torch.backends.cudnn.deterministic = True



class Logger:
    def __init__(self, file_log_path, tb_log_path=None):
        """
        Custom logger with the ability of logging to console, file and Tensorboard.
        """
        # Make sure log_file_name exists since logging throws error if it does not exist
        file_log_path = file_log_path + '.log'
        if file_log_path is not None:
            os.makedirs(os.path.dirname(file_log_path), exist_ok=True)

        # Set global logging level (i.e. which log levels will be ignored)
        # logging.basicConfig(level=logging.DEBUG)
        # logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

        # Create a custom logger
        self.file_logger = logging.getLogger('MyFileLogger')
        self.console_logger = logging.getLogger('MyConsoleLogger')

        # Create handlers
        c_handler = logging.StreamHandler() # writes logs to stdout (console)
        f_handler = logging.FileHandler(file_log_path, mode='w') # writes logs to a file
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        # c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_format = logging.Formatter('%(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the corresponding loggers
        self.console_logger.addHandler(c_handler)
        self.file_logger.addHandler(f_handler)

        self.console_logger.setLevel(logging.DEBUG)
        self.file_logger.setLevel(logging.DEBUG)

        # stop propagting to root logger. Refer to: https://stackoverflow.com/questions/17745914/python-logging-module-is-printing-lines-multiple-times
        self.console_logger.propagate = False
        self.file_logger.propagate = False


        # Initialize TensorBoard --------------------------------------------------------------------
        if tb_log_path is not None:
            # run_name =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tb_summaryWriter = SummaryWriter(tb_log_path)

            self.tb_global_steps = defaultdict(init_tb_global_step)
        
        self.GLOBAL_RANK = None
        
    def log_msg_to_console(self, msg):
        """
        Passed in message will be logged to console.
        """
        mgs = '\n' + msg + '\n'
        self.console_logger.info(msg)

    def log_msg_to_file(self, msg):
        """
        Passed in message will be logged to file.
        """
        mgs = '\n' + msg + '\n'
        self.file_logger.info(msg)
    
    def log_msg_to_both_console_and_file(self, msg, only_rank_0 =False):
        """
        Passed in message will be logged to both console and file.
        """
        def log():
            mgs = '\n' + msg + '\n'
            self.console_logger.info(msg)
            self.file_logger.info(msg)

        if only_rank_0:
            if self.GLOBAL_RANK == 0:
                log()
        else:
            log()

    def log_dict_to_file(self, info_dict):
        """
        key and value pairs in the info_dict will be logged into the file.
        """
        log_str = '\n'
        log_str += '='*20 + '\n'
        for k,v in info_dict.items():
            log_str += f'{k}: {v} \n'
        log_str += '='*20
        log_str += '\n'
        self.file_logger.info(log_str)

    def log_to_file(self, entity):
        log_str = '\n'
        log_str += '='*20 + '\n'
        log_str += str(entity) +'\n'
        log_str += '='*20
        log_str += '\n'
        self.file_logger.info(log_str)
    
    def log_scalar_to_tb_without_step(self, tag, scalar_value, only_rank_0=True):
        def log():
            global_step = self.tb_global_steps[tag] # get current global_step for the tag
            self.tb_summaryWriter.add_scalar(tag, scalar_value, global_step) # log scalar to tb
            self.tb_global_steps[tag] = self.tb_global_steps[tag] + 1 # update global_step for the tag

        if only_rank_0:
            if self.GLOBAL_RANK == 0:
                log()
        else:
            log()

    def log_scalar_to_tb_with_step(self, tag, scalar_value, step, only_rank_0=False):
        if only_rank_0:
            if self.GLOBAL_RANK == 0:
                self.tb_summaryWriter.add_scalar(tag, scalar_value, step) # log scalar to tb
        else:
            self.tb_summaryWriter.add_scalar(tag, scalar_value, step) # log scalar to tb
    
    def log_gradients_in_model_to_tb_without_step(self, model, model_name, log_full_detail=False, only_rank_0=True):
        """
        Logs information (grads, means, ...) about the the parameters of the given model to tensorboard.
        Inputs:
            model (torch.nn.Module): model to log its paramters
            model_name (str): information will be logged under the given model_name in tensorboard
            log_full_detail (bool): if False, just logs norm of the overall grdients. If True, logs more detailed info per weights and biases.
        """
        def log():
            tag = f'log_gradients_in_model_{model_name}'
            global_step = self.tb_global_steps[tag] # get current global_step for the tag

            all_weight_grads = torch.tensor([])
            all_bias_grads = torch.tensor([])
            # Log gradients to Tensorboard
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                if "weight" in name: # Model weight
                    if log_full_detail:
                        self.tb_summaryWriter.add_histogram("grad/"+model_name+"/"+name, param.grad, global_step)
                        self.tb_summaryWriter.add_scalar("grad.mean/"+model_name+"/"+name, param.grad.mean(), global_step)

                    all_weight_grads = torch.concat([all_weight_grads, param.grad.cpu().reshape(-1)])

                elif "bias" in name: # Model bias
                    if log_full_detail:
                        self.tb_summaryWriter.add_histogram("grad/"+model_name+"/"+name, param.grad, global_step)
                        self.tb_summaryWriter.add_scalar("grad.mean/"+model_name+"/"+name, param.grad.mean(), global_step)

                    all_bias_grads = torch.concat([all_bias_grads, param.grad.cpu().reshape(-1)])
        
            # Log norm of all the model grads concatenated together to form one giant vector
            all_weight_grads_norm = torch.norm(all_weight_grads, 2)
            all_bias_grads_norm = torch.norm(all_bias_grads, 2)
            self.tb_summaryWriter.add_scalar("all_weight_grads_norm/"+model_name, all_weight_grads_norm.item(), global_step)
            self.tb_summaryWriter.add_scalar("all_bias_grads_norm/"+model_name, all_bias_grads_norm.item(), global_step)

            self.tb_global_steps[tag] = self.tb_global_steps[tag] + 1 # update global_step for the tag

        if only_rank_0:
            if self.GLOBAL_RANK == 0:
                log()
        else:
            log()

    def log_parameters_in_model_to_tb_without_step(self, model, model_name, only_rank_0=False):
        """
        Logs information (weights, biases, means, ...) about the the parameters of the given model to tensorboard.
        Inputs:
            model (torch.nn.Module): model to log its paramters
            model_name (str): information will be logged under the given model_name in tensorboard
            log_full_detail (bool): if False, just logs norm of the overall grdients. If True, logs more detailed info per weights and biases.
        """
        def log():
            tag = f'log_parameters_in_model_{model_name}'
            global_step = self.tb_global_steps[tag] # get current global_step for the tag

            # Log weights and biases to Tensorboard
            for name, param in model.named_parameters():
                if "weight" in name: # Model weight
                    self.tb_summaryWriter.add_histogram("weight/"+model_name+"/"+name, param, global_step)
                    self.tb_summaryWriter.add_scalar("weight.mean/"+model_name+"/"+name, param.mean(), global_step)

                elif "bias" in name: # Model bias
                    self.tb_summaryWriter.add_histogram("bias/"+model_name+"/"+name, param, global_step)
                    self.tb_summaryWriter.add_scalar("bias.mean/"+model_name+"/"+name, param.mean(), global_step)

            self.tb_global_steps[tag] = self.tb_global_steps[tag] + 1 # update global_step for the tag

        if only_rank_0:
            if self.GLOBAL_RANK == 0:
                log()
        else:
            log()
        

class ParallelizedEnvironmentRenderer:
    def __init__(self, num_envs, figsize=(6, 8)):
        self.num_envs = num_envs
        self.fig, self.axs = plt.subplots(num_envs, figsize=figsize, constrained_layout=True)
        plt.ion()

    def render(self, state):
        for i in range(self.num_envs):
            self.axs[i].imshow(state[i].squeeze(0).astype(np.uint8), cmap="gray")
            self.axs[i].set_title(f'env: {i}')
            self.axs[i].tick_params(top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False)
        plt.pause(1/60)
    
    def close(self):
        plt.close()


def print_config_options():
    print(
        """
***********************************************
***********************************************
        [DEFAULT]
        -TrainMethod: currently does not make a difference. Keep it as 'RND'
        -representationLearningMethod: can be chosen from [None, BYOL, Barlow-Twins]. 'None' means just RND. 'BYOL' means use RND and BYOL combined. 'Barlow-Twins' means using RND and Barlow Twins methods combined.



        -EnvType: can be chosen from [atari, mario, classic_control]. Specifies the category that openai gym environment belongs to. An Environment Process instance is created according to its value.
        -EnvID: specifies the argument givent to gym.make(<EnvID>). Examples include [MontezumaRevengeNoFrameskip-v4, SuperMarioBros-v0, PongNoFrameskip-v4, CartPole-v1].


        #------
        -Epoch: number of epochs to perform optimization with a given rollout
        -MiniBatch: number of minibatches to divide the training data into
        -LearningRate: learning rate used by the optimizer


        # PPO ->
        -PPOEps: PPO clip is calculated as surr2 = clamp(ratio, 1 - PPOEps, 1 + PPOEps)
        -Entropy: # entropy coefficient (loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + rnd_loss + self.representation_loss_coef * representation_loss).

        # ------ Exploration
        # RND ->
        -NumEnv: number of parallel environments (i.e. number of parallel processes that runs different environments simulataneously).
        -NumStep: this many steps are taken in every 'NumEnv' many parallel environments before updating the model.
        -MaxStepPerEpisode: maximum number of steps that can be taken per episode.
        -LifeDone: (bool) is related to SuperMarioBros gym environment. When 'True', then an episode ends when Mario looses a life.
        -StateStackSize: number of past frames (images) stacked to create state representation (i.e. history size).
        -StickyAction: if 'True', then sticky actions (last action is repeated) are taken in gym environments with probability of ActionProb.
        -ActionProb: probability used for sticky actions. See 'StrickAction' parameter explanation.
        -IntGamma: gamma used for calculating the Return for intrinsic rewards (i.e. R_i = sum_over_t((intrinsic_gamma ** t) * intrinsic_reward_t)) (i.e. future reward discount factor).
        -Gamma: gamma used for calculating the Return for extrinsic rewards (i.e. R_e = sum_over_t((intrinsic_gamma ** t) * extrinsic_reward_t) (i.e. future reward discount factor).
        -ExtCoef: coefficient of extrinsic reward in the calculation of Combined Advantage (i.e. A = (A_i * IntCoef) + (A_e * ExtCoef).
        -IntCoef: coefficient of intrinsic reward in the calculation of Combined Advantage (i.e. A = (A_i * IntCoef) + (A_e * ExtCoef).
        -UpdateProportion: proportion of experience used for training the predictor network in RND (refer to RND paper for an explanation).
        -UseGAE: if 'True', then GAE (Generalized Advantage Estimation) is used for advantage calculation.
        -GAELambda: lambda in GAE. See 'UseGAE' parameter explanation
        -PreProcHeight: height of the images (frames) used as the observations in the gym environments after preprocessing them.
        -ProProcWidth: width of the images (frames) used as the observations in the gym environments after preprocessing them.
        -ObsNormStep: (numStep * ObsNormStep) number of initial steps are taken for initializing observation normalization parameters.
        -UseNoisyNet: if 'True', then noisy Linear layers are used instead of regular Linear (Dense) layers in the CNN network.

        # CNN Actor-Critic dims (from RND): refer to https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py


        # ------ Representation Learning

        -apply_same_transform_to_batch: if 'False', then a new transformation (used for augmenting stacked states) is sampled per each element in the batch, otherwise ('True') only one transformation is sampled per batch.

        # BYOL->
        -BYOL_projectionHiddenSize: original implementation on ImageNet used '4096'. Refers to the projection size of the hidden layer in BYOL network.
        -BYOL_projectionSize: original implementation on ImageNet used '256'. Refers to the projection size (embedding size) of the projection layer in BYOL network.
        -BYOL_movingAverageDecay: original implementation on ImageNet used a  dynamically changing value.
        -BYOL_representationLossCoef: BYOL loss is multiplied with this coefficient (i.e. loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + rnd_loss + self.representation_loss_coef * representation_loss).

        # Barlow-Twins ->
        -BarlowTwinsLambda: trade-off parameter lambda of the Barlow-Twins loss function (loss = on_diag + self.lambd * off_diag).
        -BarlowTwinsProjectionSizes: original implementation on ImageNet used [8192, 8192, 8192]. Specifies the dimensions of the Linear layers of the projector network for Barlow-Twins.
        -BarlowTwins_representationLossCoef: Barlow-Twins loss is multiplied with this coefficient (loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy + rnd_loss + self.representation_loss_coef * representation_loss).

        -# ------



        -loadModel: if 'True', a training checkpoint (model, optimizer, epoch, ...) is loaded from the path specified by the command line argument passed to '--load_model_path'.
        -render: if 'True', then environments are rendered simultaneously on a new window. During training (i.e. '--train') do not set this to 'True' since multiple environments are playing out simultaneously. Set to 'True' during evaluation mode (i.e. '--eval').
        -saveCkptEvery: after every this many episodes during training a checkpoint is saved.
        -StableEps = 1e-8
        -UseGPU: if 'True' then GPU (cuda) is used, else CPU.
        -UseGradClipping: (bool) If True, then gradient clipping (max grad norm = 'MaxGradNorm') is used.
        -MaxGradNorm: (float) If 'UseGradClipping' is True, then this value specifies the maximum norm of the gradient which will not be clipped.


        -[OPTIONS]
        -EnvType = [atari, mario, classic_control]
***********************************************
***********************************************
        """
        )