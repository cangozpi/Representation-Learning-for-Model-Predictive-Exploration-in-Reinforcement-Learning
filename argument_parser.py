import argparse
from os import path
import datetime

def get_args():
    """
    defines an argument parser and returns the passed in arguments as a dict.
    Return:
        args (dict): parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate RND agents with auxiliary representation learning losses.")


    parser.add_argument("--seed", type=int, default=42, help="seed used for seeding torch, numpy, random, gym for reproducibility.")
    parser.add_argument("--config_path", type=str, default=path.join('.', 'configs', 'MontezumaRevenge', 'config_rnd00.conf'), help="relative path for the config.conf file to use.")
    parser.add_argument("--log_name", type=str, default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), help="name used for naming both the log file and the Tensorboard log.")
    parser.add_argument("--train", type=bool, default=False, const=True, nargs='?', help="When specified, the agent will be trained.")
    parser.add_argument("--eval", type=bool, default=False, const=True, nargs='?', help="When specified, the agent will be evaluated (tested).")
    parser.add_argument("--load_model_path", type=str, default='./checkpoints/ckpt00.ckpt', help="if \'load_model\' is set to True in the config file, then the corresponding model (agent) will be loaded from the path specified by the value of this argument.")
    parser.add_argument("--save_model_path", type=str, default='./checkpoints/ckpt00.ckpt', help="Checkpoints will be saved during the model (agent) training to the path specified by this argument.")
    parser.add_argument("--config_options", type=bool, default=False, const=True, nargs='?', help="Prints explanations of available parameters in config.conf files (see \'--config-path\').")
    parser.add_argument("--pytorch_profiling", type=bool, default=False, const=True, nargs='?', help="Uses pytorch profiler and saves the logs at 'logs/torch_profiler_logs' directory.")

    # args = vars(parser.parse_args())

    # Options in 'unknown' array should not be used by the program. They are here to prevent errors that appear when using scalene to profile
    # torchrun (python -m torch.distributed.run). When using scalene the options passed to torchrun gets passed to main.py program and our
    # argument parser crashes due to torchrun's commands not being defined for our parser above. The following line prevents such errors
    # by encapsulating argument options which are passed to our program but were not defined above by us into the variable 'unknown' (type: array).
    args, unknown = parser.parse_known_args() # --> dict, array (see: https://stackoverflow.com/questions/12818146/python-argparse-ignore-unrecognised-arguments)
    args = vars(args)
    assert args['train'] != args['eval'], "cannot supply both \'--train\' and \'--eval\' options at the same time."

    return args