# RLearn
from rlearn.algorithms import Agent
from rlearn.implemented_agents import make_agent
from rlearn.helper import get_configs, get_hp_dict

# Other
import gym
import argparse
import keyboard
from sys import exit
import json
from yaml import SafeLoader
import wandb


def get_args():
    parser = argparse.ArgumentParser(
                    prog = 'python run.py',
                    description = 'Train or eval an agent on an env.',
                    epilog = 'Example: python run.py --agent DQN --env CartPole-v1 --steps 10000 --episodes 1000 --wandb_cb True --n_render 20'
                    )
    # Agent and env
    parser.add_argument('--agent', type = str, help = 'Agent to use', required = True)
    parser.add_argument('--env', type = str, help = 'Env to use', required = True)
    parser.add_argument('--train', type = bool, help = 'Whether to train or eval', default = True)
    parser.add_argument('--load', type = bool, help = 'Whether to load a model', default = None)
    # Configs
    parser.add_argument('--agent-cfg', type = str, help = 'Path to agent config file', default = None)
    parser.add_argument('--inter-cfg', type = str, help = 'Path to interface config file', default = None)
    parser.add_argument('--train-cfg', type = str, help = 'Path to training config file', default = None)
    # Logging
    parser.add_argument('--wandb', type = bool, help = 'Whether metrics are logged in WandB', default = None)
    parser.add_argument('--tb', type = bool, help = 'Whether metrics are logged in Tensorboard', default = None)
    parser.add_argument('--n_render', type = int, help = 'One episode on n_render is rendered', default = None)
    # Other hyperparams
    parser.add_argument('--hp', type = str, help = 'Dict of other hyperparameters {hp_name: value} with value being 0.9, SARSA, None, SARSA for example', default = None)

    args = parser.parse_args()

    # Add custom hyperparameters from --hp arg
    if args.hp is not None:
        hp_dict = get_hp_dict(args.hp)
        for key, value in hp_dict.items():
            setattr(args, key, value)
        print(f'Overriding args hyperparameters with {hp_dict}')

    return args
    
    


if __name__ == '__main__':
    args = get_args()
    
    print("Run starts. Press 'q' to quit.")

    # Get configs from config paths
    agent_cfg, train_cfg = get_configs(args)
    # Override configs with CLI args
    for attr in vars(args):
        if attr in agent_cfg and getattr(args, attr) is not None:
            agent_cfg[attr] = getattr(args, attr)
            print(f'Overriding {attr} in agent config with {getattr(args, attr)}')
        if attr in train_cfg and getattr(args, attr) is not None:
            train_cfg[attr] = getattr(args, attr)
            print(f'Overriding {attr} in training config with {getattr(args, attr)}')
    # None values to inf values
    train_cfg["episodes"] = train_cfg["episodes"] if train_cfg["episodes"] is not None else float('inf')
    train_cfg["steps"] = train_cfg["steps"] if train_cfg["steps"] is not None else float('inf')
    train_cfg["n_render"] = train_cfg["n_render"] if train_cfg["n_render"] is not None else float('inf')
    # Get some training/logging variables
    steps = train_cfg["steps"]
    episodes = train_cfg["episodes"]
    do_wandb = train_cfg['wandb']
    do_tb = train_cfg['tb']
    n_render = train_cfg['n_render']
    


    # Create env and agent
    env : gym.Env = gym.make(args.env)
    agent : Agent = make_agent(
        agent_name = args.agent, 
        env = env, 
        agent_cfg = agent_cfg,
        train_cfg=train_cfg,
        )
    

    # Logging
    if do_wandb:
        try:
            from settings import project, entity
        except ImportError:
            raise Exception("You need to specify your WandB ids in settings.py\nTemplate is available at div/settings_template.py")
        train_run = wandb.init(project = project,
                        entity = entity,
                        config = agent.config,)

    # Load model
    # if load: pass
    
    # Training
    episode = 1
    step = 0
    while step < steps and episode < episodes:
        done = False
        obs = env.reset()
        while not done and step < steps and episode < episodes:
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
            if episode % n_render == 0:
                env.render()
            agent.log_metrics()
            if do_tb:
                pass
            if keyboard.is_pressed('q'):
                print("\nQuitting...")
                exit()

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













