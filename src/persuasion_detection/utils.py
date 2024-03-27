import torch
import os
import argparse
import yaml
import logging
import sys
import time
import dill as pickle
import pandas as pd 
from transformers import AutoTokenizer
import sys 

sys.path.append('/Users/rt853/repos/UoB/bath-persuasion-detection/src/persuasion_detection')

from persuasion_strategy_dataset import PersuasionStrategyDataset
from torch.utils.data import DataLoader

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
    
    model = BertClassifier(
        'bert',
        config.get('pretrained_model'),
        
        num_classes)
    
    print('Loading model from: ', config.get('testing_model_path'))
    
    model.load_state_dict(
        torch.load(
            config.get('testing_model_path'),
            map_location=torch.device('cpu')
        )
    )
    
    model.eval()
    return model


def load_dataloader(config):

    if config.get('eval_dataloader_path')[-4:] == '.pkl':
        with open(
            f"{config.get('eval_dataloader_path')}",
            'rb'
                ) as f:
            dataloader = pickle.load(f)
    
    elif config.get('eval_dataloader_path')[-4:] == '.csv':
        df = pd.read_csv(config.get('eval_dataloader_path'))
        
        if config.get('drop_social_pressure'):
            
            df = df.drop(columns=['6-SOCIAL', '7-PRESSURE'])
            
        pdataset = PersuasionStrategyDataset(
            df,
            AutoTokenizer.from_pretrained(config.get('pretrained_model')),
            max_token_len=config.get('max_token_len')
        )
        
        dataloader = DataLoader(
            pdataset,
            batch_size=32,
        )

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


def create_filename(hyperparameters):
    current_time = time.strftime("%Y%m%d-%H%M%S")
    hyperparameters_str = "_".join(
        f"{k}={v}" for k, v in hyperparameters.items()
        )

    filename = f"{hyperparameters_str}_{current_time}.txt"

    return filename
