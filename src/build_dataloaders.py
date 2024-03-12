from collections import Counter
import torch
import logging
from torch.utils.data import WeightedRandomSampler, DataLoader
from transformers import AutoTokenizer

from persuasion_strategy_dataset import PersuasionStrategyDatasetBERT
from persuasion_strategy_dataset import PersuasionStrategyDatasetLSTM
from utils import load_config, get_args, output_pickle, gen_metadata
from preprocess import import_data


def build_sampler(training_dataset):
    """
    Build a weighted random sampler to balance the dataset during training.

    Args:
        training_dataset (Dataset): The training dataset.

    Returns:
        WeightedRandomSampler: The weighted random sampler.

    """
    labels = [torch.argmax(label['labels']).item()
              for label in training_dataset]
    class_distribution = Counter(labels)

    # Calculate weights for each sample
    class_weights = {class_label: len(training_dataset) /
                     (len(class_distribution) * class_count) for
                     class_label, class_count in class_distribution.items()}

    weights = [class_weights[label] for label in labels]

    # Convert weights to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float)

    # Create a sampler to balance the dataset during training
    sampler = WeightedRandomSampler(weights_tensor, len(weights_tensor))
    return sampler


def gen_dataloader(df, tokenizer, config, vocab):
    """
    Generate a DataLoader object from a DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the data.
        tokenizer (BertTokenizer): The tokenizer to use for encoding the data.
        max_token_len (int): The maximum token length for the input data.
        batch_size (int): The batch size for the DataLoader.
        sampler (WeightedRandomSampler): The sampler to use for the DataLoader.

    Returns:
        DataLoader: The DataLoader object.

    """

    if config.get('use_pretrained'):
        dataset = PersuasionStrategyDatasetBERT(
            data=df,
            tokenizer=tokenizer,
            max_token_len=config.get('max_token_len')
        )

    else:
        dataset = PersuasionStrategyDatasetLSTM(
            data=df,
            vocab=vocab
        )

    if config.get('use_sampler'):
        sampler = build_sampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=config.get('batch_size'),
            sampler=sampler
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.get('batch_size')
        )

    return dataloader


def build_dataloaders(train_df, test_df, config, vocab):
    """
    Build dataloaders for training and testing data.

    Args:
        train_df (pandas.DataFrame): The training dataset.
        test_df (pandas.DataFrame): The testing dataset.
        config (dict): A dictionary containing configuration parameters.

    Returns:
        tuple: A tuple containing the training
        dataloader and testing dataloader.
    """

    tokenizer = None
    logger = logging.getLogger(__name__)
    if config.get('use_pretrained'):
        tokenizer = AutoTokenizer.from_pretrained(
            config.get('pretrained_model'))

    train_dataloader = gen_dataloader(
        train_df,
        tokenizer,
        config,
        vocab)

    if config.get('output_dataloader'):
        metadata = gen_metadata(config, 'dataloader')
        output_pickle(train_dataloader, config,
                      'training_dataloader_{}'.format(metadata) + '.pkl')

    test_dataloader = gen_dataloader(
        test_df,
        tokenizer,
        config,
        vocab
    )

    if config.get('output_dataloader'):
        metadata = gen_metadata(config, 'dataloader')
        output_pickle(test_dataloader, config,
                      'testing_dataloader_{}'.format(metadata) + '.pkl')

    logger.info('Dataloaders built')
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    # Example usage
    args = get_args()
    config = load_config(args.config)
    train_df, test_df, label_names = import_data(config)
    training_dataloader, testing_dataloader = build_dataloaders(
        train_df, test_df, config
    )
