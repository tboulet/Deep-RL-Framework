import argparse
import yaml
from yaml import SafeLoader

try:
    from settings import config_type
except ImportError:
    config_type = 'default'
    print('No settings.py file found. Using default config. You can create a settings.py file following templates/settings.py.')


def get_configs(args : argparse.Namespace):
    if args.agent_cfg is None:
        agent_cfg_path = f'configs/{config_type}/algorithms/{args.agent}_config.yaml'
    else:
        agent_cfg_path = args.agent_cfg
    agent_cfg = yaml.load(open(agent_cfg_path, 'r'), Loader = SafeLoader)
    agent_cfg = agent_cfg if agent_cfg is not None else {}

    if args.train_cfg is None:
        train_cfg_path = f'configs/{config_type}/train_config.yaml'
    else:
        train_cfg_path = args.train_cfg
    train_cfg = yaml.load(open(train_cfg_path, 'r'), Loader = SafeLoader)
    train_cfg = train_cfg if train_cfg is not None else {}

    return agent_cfg, train_cfg

def get_hp_dict(hp_str : str):
    hp_dict = {}
    # Assert good format
    if len(hp_str) <= 2 or hp_str[0] != '{' or hp_str[-1] != '}':
        raise Exception(f'Invalid hp string: {hp_str}')
    # Remove brackets and iterate over pairs
    for hp in hp_str[1:-1].split(','):
        hp_name, hp_value = hp.split(':')
        # Convert to value
        if is_float_str(hp_value):
            hp_value = float(hp_value)
        if is_bool_str(hp_value):
            hp_value = str_to_bool(hp_value)
        hp_dict[hp_name] = hp_value
    return hp_dict




def is_float_str(s : str):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_bool_str(s : str):
    return s in ['True', 'true', '1', 'False', 'false', '0']

def str_to_bool(s : str):
    if s in ['True', 'true', '1']:
        return True
    elif s in ['False', 'false', '0']:
        return False
    else:
        raise Exception(f'Invalid boolean string: {s}')
