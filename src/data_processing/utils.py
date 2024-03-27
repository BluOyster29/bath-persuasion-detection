import argparse
import yaml
import logging


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
