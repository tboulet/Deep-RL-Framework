import gym

def create_gym_env(id : str, kwargs : dict = {}):
    """
    Create a gym environment with a given id and kwargs
    """
    env = gym.make(id, **kwargs)
    return env