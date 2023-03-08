# RLearn
from typing import Callable
from rlearn.agents import Agent
from rlearn.core.helper import string_to_callable, choose_model_to_load, create_model_path, set_random_seed, none_to_infs, none_to_empty_dict

# Configs
import hydra
from omegaconf import DictConfig, OmegaConf

# Other
import gym
from time import sleep

    
    

@hydra.main(config_path = 'configs', config_name = 'train_config')
def main(config : DictConfig):
    print("\nTraining with config :\n", OmegaConf.to_yaml(config))
    
    # Training & eval numerical parameters
    train_timesteps = config.training.train_timesteps
    train_episodes = config.training.train_episodes
    eval_freq = config.training.eval_freq
    render_freq = config.training.render_freq
    n_eval_episodes = config.training.n_eval_episodes
    train_timesteps, train_episodes, eval_freq, render_freq, n_eval_episodes = none_to_infs(train_timesteps, train_episodes, eval_freq, render_freq, n_eval_episodes)
    config.algo.algo_config, config.env.env_kwargs, config.training.metrics, config.training.loggers = none_to_empty_dict(config.algo.algo_config, config.env.env_kwargs, config.training.metrics, config.training.loggers)
    # Loading 
    checkpoint = config.training.checkpoint
    checkpoint_criteria = config.training.checkpoint_criteria

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


    # Create env
    create_env_fn = string_to_callable(config.env.create_env_fn_string)
    env_kwargs = config.env.env_kwargs
    env : gym.Env = create_env_fn(**env_kwargs)

    # Create agent
    algo_class_string = config.algo.create_algo_fn_string
    algo_class = string_to_callable(algo_class_string)
    agent : Agent = algo_class(
        env = env, 
        config = config,)

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









