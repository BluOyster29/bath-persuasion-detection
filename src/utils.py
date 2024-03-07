import yaml
import argparse
import pickle

def getargs():
    p = argparse.ArgumentParser()
    p.add_argument('config_path', type=str, default='.config.yaml')
    return p.parse_args()


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def unpickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)
    
def convert_config_to_text(config):
    """
    Converts a dictionary of configuration options to a formatted text output.

    Args:
        config (dict): A dictionary containing configuration options.

    Returns:
        str: A formatted text output representing the configuration options.
    """
    text_output = ""
    for key, value in config.items():
        text_output += f"{key}: {value}\n"
    return text_output

