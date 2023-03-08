import gym

def create_gym_env(id : str, kwargs : dict = {}) -> gym.Env:
    """Create a gym environment with a given id and kwargs

    Args:
        id (str): the id of the environment
        kwargs (dict, optional): the kwargs to pass to the environment. Defaults to {}.
    """
    env = gym.make(id, **kwargs)
    return env