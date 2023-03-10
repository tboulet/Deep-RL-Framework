# Metrics are object with methods called inside every RL agent (after acting, remembering and learning).
# They log efficiently information using WandB and can be easily defined.

from abc import ABC, abstractmethod
from numbers import Number
import os
from time import time
from typing import Dict, List, Type

import wandb
from torch.utils.tensorboard import SummaryWriter



class Logger(ABC):
    """This is the base class for all loggers.
    """
    def __init__(
        self,
        project_name : str,
        run_name : str,
        run_config : dict,
        log_dir : str,
    ):
        """The logger is used for logging metrics during training.

        Args:
            project_name (str): the name of the project.
            run_name (str): the name of the run / experiment.
            run_config (dict): the configuration of the run / experiment.
            log_dir (str): the directory where the logs will be saved in local
        """
        pass

    @abstractmethod
    def log_metrics(self,
            list_of_metrics : Dict[str, Number],
            step : int,
            ) -> None:
        """Log a dictionary of metrics as having been computed at a given step.

        Args:
            list_of_metrics (List[Dict[str, Number]]): a list of dictionaries of metrics (e.g. [{"loss": 0.5, "accuracy": 0.8}] ).
            step (int): the number of timesteps since the beggining of the training at which the metrics were computed.
        """
        pass



class ConsoleLogger(Logger):
    """A logger that print metrics to the console. Mainly for debugging purposes."""
    name = "ConsoleLogger"
    def __init__(self, **kwargs):
        pass
    def log_metrics(self,
            list_of_metrics : List[Dict[str, Number]],
            step : int,
            ) -> None:
        for metrics in list_of_metrics:
            if len(metrics) > 0:
                print(f"Step {step} : {metrics}")



class TensorboardLogger(Logger):
    """ A logger that log metrics to tensorboard."""
    name = "TensorboardLogger"
    def __init__(
        self,
        project_name : str,
        run_name : str,
        run_config : dict,
        log_dir : str,
    ):
        log_dir = os.path.join(log_dir, "tensorboard", run_name)
        self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
    def log_metrics(self,
            list_of_metrics : List[Dict[str, Number]],
            step : int,
            ) -> None:
        for metrics in list_of_metrics:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step)



class WandbLogger(Logger):
    """ A logger that log metrics to wandb."""
    name = "WandbLogger"
    def __init__(
        self,
        project_name : str,
        run_name : str,
        run_config : dict,
        log_dir : str,
    ):
        super().__init__()
        self.run = wandb.init(
            project = project_name,
            name = run_name,
            config=run_config,
            dir = f"{log_dir}/wandb/{run_name}",
            sync_tensorboard=True,  
            monitor_gym=True,  # TODO : add monitor_gym
            save_code=True, 
            )
    def log_metrics(self,
            list_of_metrics : List[Dict[str, Number]],
            step : int,
            ):
        for metrics in list_of_metrics:
            wandb.log(metrics, step = step)



logger_names_to_classes = {
    ConsoleLogger.name : ConsoleLogger,
    TensorboardLogger.name : TensorboardLogger,
    WandbLogger.name : WandbLogger,
}

def get_loggers_classes(logger_names : List[str]) -> List[Type[Logger]]:
    """Function that returns a list of loggers classes from a list of logger names as they are defined in the config file.

    Args:
        logger_names (List[str]): the list of logger names as they are defined in the config file.

    Returns:
        List[Type[Logger]]: a list of logger classes.
    """
    if logger_names is None:
        return []
    logger_list = []
    for logger_name in logger_names:
        try:
            logger_list.append(logger_names_to_classes[logger_name])
        except KeyError:
            print(f"WARNING : Logger {logger_name} not found in available loggers.")
    return logger_list