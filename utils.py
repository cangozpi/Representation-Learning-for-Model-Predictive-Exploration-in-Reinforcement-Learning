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

# train_method = default_config['TrainMethod']
# if default_config['TrainMethod'] in ['PPO', 'ICM', 'RND']:
#     num_step = int(ppo_config['NumStep'])
# else:
#     num_step = int(default_config['NumStep'])

use_gae = default_config.getboolean('UseGAE')
GAE_lambda = float(default_config['GAELambda'])


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

        # Initialize TensorBoard --------------------------------------------------------------------
        if tb_log_path is not None:
            # log_file_dir = '/'.join(log_file_name.split('/')[:-1])
            run_name =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tb_summaryWriter = SummaryWriter(tb_log_path)

            self.tb_global_steps = defaultdict(lambda : 0)

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
    
    def log_msg_to_both_console_and_file(self, msg):
        """
        Passed in message will be logged to both console and file.
        """
        mgs = '\n' + msg + '\n'
        self.console_logger.info(msg)
        self.file_logger.info(msg)

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
    
    def log_scalar_to_tb_without_step(self, tag, scalar_value):
        global_step = self.tb_global_steps[tag] # get current global_step for the tag
        self.tb_summaryWriter.add_scalar(tag, scalar_value, global_step) # log scalar to tb
        self.tb_global_steps[tag] = self.tb_global_steps[tag] + 1 # update global_step for the tag

    def log_scalar_to_tb_with_step(self, tag, scalar_value, step):
        self.tb_summaryWriter.add_scalar(tag, scalar_value, step) # log scalar to tb

