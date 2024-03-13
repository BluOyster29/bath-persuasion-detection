from torch import load
import os
import argparse
import yaml
import pickle
import logging
import sys

module_dir = os.path.expanduser(
    '~/repos/UoB/bath-persuasion-detection/models/')
sys.path.append(module_dir)

from bert_classifier import BertClassifier # noqa


def get_args():
    # Generate Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config.yaml")
    return parser.parse_args()


def load_config(config_path):
    # Load Config

    logger = logging.getLogger(__name__)
    logger.info('Loading config from %s', config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def output_pickle(object, config, item_name):
    # Output Pickle
    with open(f"{config.get('output_dataloader')}_{item_name}.pkl", 'wb') as f:
        pickle.dump(object, f)


def gen_metadata(config, metatype, **kwargs):
    """
    Generate metadata based on the given configuration and metatype.

    Args:
        config (dict): A dictionary containing configuration parameters.
        metatype (str): The type of metadata to generate.

    Returns:
        str: The generated metadata.

    """
    if metatype == 'datalodaer':
        meta = f"{config.get('batch_size')}_{config.get('max_token_len')}_"
        meta += f"{config.get('sampler')}"

    if metatype == 'model':
        meta = f"{config.get('pretrained_model')}_{config.get('num_epochs')}_"
        meta += f"{config.get('learning_rate')}_{config.get('batch_size')}"
    return meta


def load_model(config, num_classes):

    model = BertClassifier(config.get('pretrained_model'), num_classes)
    model.load_state_dict(load(config.get('model_path')))
    return model


def load_dataloader(config, item_name):
    with open(
        f"{config.get('eval_dataloader_path')}.pkl",
        'rb'
            ) as f:
        dataloader = pickle.load(f)
    return dataloader


def output_predictions(predictions, config):

    with open(config.get('output_predictions_path'), 'w') as f:
        for prediction in predictions:
            pred, true = prediction
            f.write(f'True: {pred} - ' + f'Pred: {true}' + '\n')


def fetch_labels(config):

    labels = [
        '1-RAPPORT',
        '2-NEGOTIATE',
        '3-EMOTION',
        '4-LOGIC',
        '5-AUTHORITY',
        '6-SOCIAL',
        '7-PRESSURE',
        '8-NO-PERSUASION'
    ]

    if config.get('drop_columns'):
        for i in config.get('drop_columns'):
            labels.remove(i)

    return labels


def open_texts(path):
    with open(path, 'r') as file:
        texts = file.read().readlines()
    return texts
