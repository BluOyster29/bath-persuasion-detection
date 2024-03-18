from utils import load_config, load_dataloader
from utils import fetch_labels
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from persuasion_strategy_dataset import PersuasionStrategyDatasetBERT
from tqdm.auto import tqdm
from evaluate import gen_stats, output_stats
import torch
import argparse
import pandas as pd
import logging
import sys
import warnings
from bert_classifier import BertClassifier # noqa

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_args():
    """
    Parse command line arguments for testing.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument('config_path', help='Path to config.yaml')
    return p.parse_args()


def list_of_phrases_to_dataloader(data):
    """
    Convert a list of phrases to a DataLoader.
    
    Args:
        data (list): A list of phrases.
    
    Returns:
        str: The concatenated phrases.
    """
    phrases = ''
    return phrases


def csv_to_dataloader(data_path, config):
    """
    Convert a CSV file to a DataLoader.
    
    Args:
        data_path (str): The path to the CSV file.
        config (dict): The configuration settings.
    
    Returns:
        torch.utils.data.DataLoader: The DataLoader object.
    """
    df = pd.read_csv(data_path)
    
    try:
        df = df.drop(columns=['6-SOCIAL', '7-PRESSURE'])
        
    except KeyError:
        df = df.drop(columns=['6-SOCIAL', '3-EMOTION#4-LOGIC', 'id'])
    
    dataset = PersuasionStrategyDatasetBERT(
        df,
        tokenizer=AutoTokenizer.from_pretrained(
            config.get('pretrained_model')),
        max_token_len=config.get('max_token_len')
    )
    logger.info('Creating DataLoader')
    return DataLoader(dataset)


def init_model(config, device, dataloader):
    """
    Initialize the persuasion detection model.
    
    Args:
        config (dict): The configuration settings.
        device (torch.device): The device to use for training.
        dataloader (torch.utils.data.DataLoader): The DataLoader object.
    
    Returns:
        tuple: A tuple containing the initialized model and the label columns.
    """
    logger.info('Initialising Model')

    labels = dataloader.dataset.label_columns

    model = BertClassifier(config.get('pretrained_model').split('-')[0],
                           config.get('pretrained_model'),
                           len(labels))
    logger.info('Loading Weights from %s', config.get('testing_model_path'))
    model.load_state_dict(torch.load(config.get('testing_model_path'),
                                     map_location=device))
    model.to(device)
    return model, labels


def test_model(config, processed_batch, data_type):
    """
    Test the persuasion detection model.
    
    Args:
        config (dict): The configuration settings.
        processed_batch (Union[list, str, torch.utils.data.DataLoader]): The processed batch of data.
        data_type (str): The type of data being tested ('batch', 'phrase', or 'dataloader').
    
    Returns:
        tuple: A tuple containing the predictions, true labels, and label columns.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, labels = init_model(config, device, processed_batch)
    model.eval()
    preds = []
    trues = []
    for b in tqdm(processed_batch, desc='Testing', file=sys.stdout):

        input_ids = b['input_ids'].to(device)
        attention_mask = b['attention_mask'].to(device)
        true = b['labels'].argmax().item()
        output = model(input_ids, attention_mask)
        prediction = torch.sigmoid(output).argmax().item()
        preds.append(prediction)
        trues.append(true)

    return preds, trues, labels


def test(args):
    """
    Test the persuasion detection model.
    
    Args:
        args (argparse.Namespace): The parsed command line arguments.
    
    Returns:
        list: The predictions made by the model.
    """
    logger.info('Testing model')
    config = load_config(args.config_path)
    testing_data = config.get('testing_data')

    if testing_data[-3:] == 'csv':
        #  the test batch is a list of phrases
        processed_batch = csv_to_dataloader(config.get('testing_data'), config)
        predictions, trues, labels = test_model(config, processed_batch, 'batch')

    elif config.get('test_phrase'):
        # the phrase is ad hoc and needs to be preprocessed
        processed_phrase = ''
        predictions = test_model(config, processed_phrase, 'phrase')

    elif config.get('testing_dataloader'):
        # import dataloader
        testing_dataloader = load_dataloader(config.get('testing_dataloader'))
        predictions = test_model(config, testing_dataloader, 'dataloader')

    stats = gen_stats(trues, predictions, labels)
    output_stats(stats, config.get('test_stat_path'))
    # output_predictions(predictions, trues, config)
    return predictions


if __name__ == '__main__':
    args = test_args()
    predictions = test(args)
