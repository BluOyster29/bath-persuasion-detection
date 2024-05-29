import argparse
import yaml
import logging
import re
from torch import zeros


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config.yaml")
    return parser.parse_args()


def load_config(config_path):

    logger = logging.getLogger(__name__)
    logger.info('Loading config from %s', config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def find_pers_type(path):

    pers = [
        '1-RAPPORT',
        '2-NEGOTIATE',
        '3-EMOTION',
        '4-LOGIC',
        '5-AUTHORITY',
        '6-SOCIAL',
        '7-PRESSURE'
    ]

    strat = None
    for i in pers:
        if re.search(i, path):
            strat = i

    if not strat:
        raise ValueError("No persuasion type found in path")
    return strat


def encode_label(label):
    empty_tensor = zeros(2)
    empty_tensor[label] += 1
    return empty_tensor
