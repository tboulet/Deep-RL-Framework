from rlearn.agents import *
import gym
from rlearn.agents.agent import Agent
from typing import Type

agents_map = {
    "random": RandomAgent,
    "DQN": DQN,
    "DDPG": DDPG,
    "REINFORCE": REINFORCE,
    "PPO": PPO,
    "AC" : AC,
}


def is_env_action_continuous(env : gym.Env) -> bool:
    """Returns True if the action space of the environment is continuous, False otherwise.
    This is for helping creating the agent's network structure.
    We assume every environment observation space is continuous. If not it can be wrapped as one.
    
    Args:
        env (gym.Env): the environment

    Returns:
        bool: whether the action space of the environment is continuous
    """
    assert isinstance(env, gym.Env), f"env must be a gym.Env, got {type(env)}"
    return isinstance(env.action_space, gym.spaces.Box)
