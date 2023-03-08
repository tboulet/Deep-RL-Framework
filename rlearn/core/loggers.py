# Metrics are object with methods called inside every RL agent (after acting, remembering and learning).
# They log efficiently information using WandB and can be easily defined.

from abc import ABC, abstractmethod
from numbers import Number
from time import time
from typing import Dict, List

import wandb
from torch.utils.tensorboard import SummaryWriter

class Logger(ABC):
    """This is the base class for all loggers.
    """
    name = "Logger"
    def __init__(
        self,
        project_name : str,
        run_name : str,
        run_config : dict,
        log_dir : str,
    ):
        pass
    @abstractmethod
    def log_metrics(self,
            items : Dict[str, Number],
            step : int,
            ) -> None:
        pass


class ConsoleLogger(Logger):
    """A logger that print metrics to the console. Mainly for debugging purposes."""
    name = "ConsoleLogger"
    def __init__(
        self,
        project_name : str,
        run_name : str,
        run_config : dict,
        log_dir : str,
    ):
        pass
    def log_metrics(self,
            items : Dict[str, Number],
            step : int,
            ) -> None:
        print(f"Step {step} : {items}")


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
        self.tensorboard_writer = SummaryWriter(log_dir=log_dir)
    def log_metrics(self,
            items : Dict[str, Number],
            step : int,
            ) -> None:
        for key, value in items.items():
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
            dir = log_dir,
            sync_tensorboard=True,  
            monitor_gym=True,  # TODO : add monitor_gym
            save_code=True, 
            )
    def log_metrics(self,
            items : Dict[str, Number],
            step : int,
            ):
        wandb.log(items, step = self.step)



logger_names_to_classes = {
    ConsoleLogger.name : ConsoleLogger,
    TensorboardLogger.name : TensorboardLogger,
    WandbLogger.name : WandbLogger,
}

def get_loggers_classes(logger_names : List[str]) -> List[Logger]:
    """Function that returns a list of loggers from a list of logger names as they are defined in the config file.

    Args:
        logger_names (List[str]): the list of logger names as they are defined in the config file.

    Returns:
        List[Logger]: a list of loggers.
    """
    if logger_names is None:
        return []
    logger_list = []
    for logger_name in logger_names:
        try:
            logger_list.append(logger_names_to_classes[logger_name])
        except KeyError:
            print(f"WARNING : Logger {logger_name} not found in available metrics.")
    return logger_list