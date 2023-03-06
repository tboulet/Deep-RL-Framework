# RLearn
import os
from typing import Callable
from rlearn.agents import Agent
from rlearn.implemented_agents import make_agent
from rlearn.helper import get_date_hour_min, string_to_callable, choose_model_to_load, create_model_path, set_random_seed
# Configs
import hydra
from omegaconf import DictConfig, OmegaConf

# Other
import gym
import argparse
import wandb
from torch.utils.tensorboard import SummaryWriter
from time import time, sleep
import random

    
    

@hydra.main(config_path = 'configs', config_name = 'train_config')
def main(config : DictConfig):
    print("\nTraining with config :\n", OmegaConf.to_yaml(config))
    
    # Training & eval numerical parameters
    train_timesteps = config.training.train_timesteps
    train_episodes = config.training.train_episodes
    eval_freq = config.training.eval_freq
    render_freq = config.training.render_freq
    n_eval_episodes = config.training.n_eval_episodes

    # Model
    # algo_class_string = config.algo.class_string

    # Loading 
    checkpoint = config.training.checkpoint
    checkpoint_criteria = config.training.checkpoint_criteria

    # Logging
    do_wandb = config.training.do_wandb
    do_tb = config.training.do_tb
    do_print = config.training.do_print
    do_dump = config.training.do_dump

    # Logging directories
    log_path = config.training.log_path
    log_path_tb = config.training.log_path_tb
    models_path = config.training.models_path
    best_model_path = config.training.best_model_path
    final_model_path = config.training.final_model_path

    # Names
    project_name = config.training.project_name
    env_name = config.env.env_name
    algo_name = config.algo.algo_name

    # Seeding
    seed = config.training.seed
    set_random_seed(seed, using_cuda=False)






    # Create env and agent
    create_env_fn = string_to_callable(config.env.create_env_fn_string)
    env_kwargs = config.env.env_kwargs
    env : gym.Env = create_env_fn(**env_kwargs)


    # Logging
    if do_wandb:
        train_run = wandb.init(
            project = project_name,
            config=dict(), # TODO : add config
            sync_tensorboard=True,  
            monitor_gym=True,  # TODO : add monitor_gym
            save_code=True, 
            dir = os.path.join('logs'),
            )
    if do_tb:
        tb_writer = SummaryWriter(log_dir = os.path.join('logs', 'tb', f'{algo_name}_{env_name}_{get_date_hour_min()}'))
    
    # Load model TODO : add load model
    if False: 
        pass
    
    # Training
    episode = 0
    step = 0
    while step < train_timesteps and episode < train_episodes:
        done = False
        obs = env.reset()
        while not done and step < train_timesteps and episode < train_episodes:
            #Agent acts
            action = agent.act(obs)    
            #Env reacts                                             
            next_obs, reward, done, info = env.step(action)
            #Agent saves previous transition in its memory                                   
            agent.remember(obs, action, reward, done, next_obs, info)    
            #Agent learns
            agent.learn()                                                
            
            #Logging
            print(f"Episode n°{episode} - Total step n°{step} ...", end = '\r')
            if episode % render_freq == 0:
                env.render()
                sleep(0.01)
            agent.log_metrics()
            
            #Reset env if episode ended, else change state
            if done:
                step += 1
                episode += 1
                break
            else:
                step += 1
                obs = next_obs

    # if wandb: run.finish()   #End wandb run.
    print("\nEnd of run.")



if __name__ == '__main__':
    main()









