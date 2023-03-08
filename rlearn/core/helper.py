import argparse
import datetime
import importlib
import os
import random
from typing import Callable, List, Union

import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from yaml import SafeLoader



time_format = '%m-%d_%Hh%Mmin'



def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed the different random generators.

    Args:
        seed (int): the seed
        using_cuda (bool): True if you need seeding for CUDA too
    """
    # Seed python RNG
    random.seed(seed)
    # Seed numpy RNG
    np.random.seed(seed)
    # seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)

    if using_cuda:
        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def is_float_str(s : str):
    """ Check if a string can be converted to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_bool_str(s : str):
    """ Check if a string can be converted to a boolean."""
    return s in ['True', 'true', '1', 'False', 'false', '0']

def str_to_bool(s : str) -> bool:
    """ Convert a string to a boolean."""
    if s in ['True', 'true', '1']:
        return True
    elif s in ['False', 'false', '0']:
        return False
    else:
        raise Exception(f'Invalid boolean string: {s}')

def none_to_infs(*args):
    """Replace None values by np.inf.
    """
    return [np.inf if arg is None else arg for arg in args]

def none_to_empty_dict(*args):
    """Replace None values by empty dict.
    """
    return [{} if arg is None else arg for arg in args]


def time_to_str(time_instant : datetime.datetime) -> str:
    """ Convert a datetime object to a string."""
    return time_instant.strftime(time_format)

def str_to_time(time_string : str) -> datetime.datetime:
    """ Convert a string to a datetime object."""
    return datetime.datetime.strptime(time_string, time_format)

def get_date_hour_min() -> str:
    """ Get the current date, hour and minute under the form "day/month_hourhmin, e.g. "12/03_15h30"""
    now = datetime.datetime.now()
    return f'{now.day}/{now.month}_{now.hour}h{now.minute}'



def string_to_callable(class_string : str) -> Callable:
    """Get a class from a string of the form "module_name.file_name:class_name"

    Args:
        class_string (str): a string of the form "module_name.file_name:class_name"

    Returns:
        Callable: the class
    """
    module_name, class_name = class_string.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)



def create_run_name(config : DictConfig) -> str:
    """From an rlearn config object, create a run name.
    The run name is defined from the algo, the env and the current date and time, as well as a random number.

    Args:
        config (DictConfig): the rlearn config object

    Returns:
        str: the run name
    """
    return f'{config.algo.algo_name}_{config.env.env_name}_{get_date_hour_min()}_{random.randint(100, 999)}'

class InfoModel:
    """A data class giving information about a model.
    """
    def __init__(self, model_path : str) -> None:
        """A data class giving information about a model.

        Args:
            model_path (str): the path to the model
        """
        self.model_path = model_path
        self.model_name = self.model_path.split('/')[-1]
        self.algo_name, self.env_name, self.time, self.timesteps, self.reward = self.model_name.split(' ')

def create_model_path(
        model_dir : str, 
        algo_name : str,
        env_name : str,
        timesteps : int, 
        reward : float,
        ) -> str:
    """Generate a model path from the model directory, the run config and the current timestep
    and mean reward of the model.

    Args:
        model_dir (str): the path to the folder where the model will be saved
        cfg (dict): the rlearn run config
        timesteps (int): the timestep (number of steps) of the model
        reward (float): the mean reward of this model

    Returns:
        str: the model name
    """
    model_path = model_dir + '/'
    model_path += algo_name + ' '
    model_path += env_name + ' '
    model_path += time_to_str(datetime.datetime.now()) + ' '
    model_path += f"t={timesteps}" + ' '
    model_path += f"r={reward:.2f}" + ' '
    model_path = model_path[:-1]
    return model_path

def choose_model_to_load(
        models_path : Union[str, List[str]], 
        env_name : Union[str, List[str]] = None,
        algo_name : Union[str, List[str]] = None,
        criteria : str = 'reward',
        ) -> str:
    """Choose a model path according to a certain criteria (default reward).
    It only searches among the models in the folder models_path with the corresponding env and algorithm.

    Args:
        models_path (str): the path(s) to the folder containing the models
        env_name (str, optional): the name(s) of the environment on which the model was trained. Defaults to None (no constraints).
        algo_name (str, optional): the name(s) of the algorithm used to train the model. Defaults to None (no constraints).
        criteria (str, optional): the criteria for choosing the model. Defaults is reward. Available criteria are: 'reward', 'timesteps' and 'time'.

    Returns:
        str: the best model path
    """
    # Get models as InfoModel objects
    models = os.listdir(models_path)
    models = [InfoModel(models_path + '/' + model) for model in models]
    # Filter models
    if env_name is not None:
        models = [model for model in models if model.env_name == env_name]
    if algo_name is not None:
        models = [model for model in models if model.algo_name == algo_name]
    # Select best model according to criteria
    if criteria == 'reward':
        models = sorted(models, key=lambda model: model.reward, reverse=True)
    elif criteria == 'timesteps':
        models = sorted(models, key=lambda model: model.timesteps, reverse=True)
    elif criteria == 'time':
        models = sorted(models, key=lambda model: str_to_time(model.time), reverse=True)
    else:
        raise ValueError(f"criteria {criteria} not understood, choose between 'reward', 'timesteps' and 'time'")
    return models[0].model_path

def try_to_load(
        model,
        algo_name : str,
        env_name : str,
        checkpoint : str,
        criteria : str,
        verbose : int = 1,
        ):
    """Try to load a model from a checkpoint directory or file.

    Args:
        model: the model that requires loading
        algo_name (str): the name of the algorithm used to train the model
        env_name (str): the name of the environment on which the model was trained
        checkpoint (str): the path to the checkpoint directory or file
        criteria (str): the criteria for choosing the model. Defaults is reward. Available criteria are: 'reward', 'timesteps' and 'time'.
        verbose (int, optional): the verbosity level. Defaults to 1.

    Returns:
        Agent: the model with the loaded parameters if possible, else the same model
    """
    if verbose >=1 : print(f"Loading checkpoint from {checkpoint} with criteria {criteria}...")
    if checkpoint is not None:
        # If dir, search model
        if os.path.isdir(checkpoint):
            if verbose >=1 : print(f"\nPicking model to load from {checkpoint}...")
            try: 
                model_path = choose_model_to_load(
                    models_path=checkpoint,
                    env_name=env_name,
                    algo_name=algo_name,
                    criteria=criteria,
                    )
                model.load(model_path)
                if verbose >=1 : print(f"Model successfully loaded : {model_path}")
            except Exception as e:
                if verbose >=1 : print(f"WARNING : Model loading failed : {e}\n -> Training from scratch")
        # If file, load model
        elif os.path.isfile(checkpoint):
            try:
                model.load(checkpoint)
                if verbose >=1 : print(f"Model successfully loaded : {checkpoint}")
            except Exception as e:
                if verbose >=1 : print(f"WARNING : Model loading failed : {e}\n -> Training from scratch")
        else:
            if verbose >=1 : print(f"WARNING : {checkpoint} is neither a file nor a directory\n -> Training from scratch")
    else:
        print("No checkpoint to load -> Training from scratch")
    return model

