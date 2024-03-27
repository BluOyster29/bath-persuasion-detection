import pandas as pd
from sklearn.model_selection import train_test_split
from utils import getargs, load_config
from persuasion_strategy_dataset import PersuasionStrategyDataset
from transformers import AutoTokenizer
from torch.utils.data import WeightedRandomSampler
import torch
from torch.utils.data import DataLoader
from collections import Counter
import pickle
import warnings
import logging

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings('ignore', category=FutureWarning)


def fetch_data(path_to_training, path_to_testing=None):

    if path_to_testing:
        df = pd.read_csv(path_to_testing)

    else:
        df = pd.concat(
            [
                pd.read_csv(path_to_training),
                pd.read_csv(path_to_testing)
                ]
            ).drop_duplicates()

    return df


def preprocess_data(df, config):

    if config.get('drop_columns'):
        df = df.drop(columns=args.config.get('drop_columns'))

    if config.get('testing'):
        return df, None

    else:

        train_df, test_df = train_test_split(
            df, test_size=config.get('test_size'),
            shuffle=True, random_state=42
            )

    return train_df, test_df


def import_data(
    config
        ):

    df = fetch_data(
        config.get('path_to_training'), config.get('path_to_testing'))
    train_df, test_df = preprocess_data(df, config)
    return train_df, test_df


def build_sampler(training_dataset):
    labels = [
        torch.argmax(label['labels']).item() for label in training_dataset]
    class_distribution = Counter(labels)

    # Calculate weights for each sample
    class_weights = {
        class_label: len(training_dataset) / (
            len(class_distribution) * class_count)
        for class_label, class_count in class_distribution.items()
        }

    weights = [class_weights[label] for label in labels]

    # Convert weights to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float)

    # Create a sampler to balance the dataset during training
    sampler = WeightedRandomSampler(weights_tensor, len(weights_tensor))
    return sampler


def gen_loaders(
    train_df,
    test_df,
    embedding,
    max_token_len,
    sampler=None,
    val_loader=None
        ):

    tokenizer = AutoTokenizer.from_pretrained(embedding)
    training_dataset = PersuasionStrategyDataset(
        train_df,
        tokenizer,
        max_token_len
    )

    if sampler:
        sampler = build_sampler(training_dataset)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=64,
        sampler=sampler,
        )

    if isinstance(test_df, pd.DataFrame):
        val_loader = DataLoader(
            PersuasionStrategyDataset(
                test_df,
                tokenizer,
                max_token_len
            ),
            batch_size=32
        )

    return training_dataloader, val_loader


def gen_dloader_paths(config, embedding, sampler_config, drop_config):

    root = config.get('output_data_path')

    if embedding == 'TODBERT/TOD-BERT-JNT-V1':
        training_path = root + 'training/train_TODBERT_sampler_' + \
            f'{sampler_config}_drop_labels_{drop_config}.pkl'
        testing_path = root + 'testing/test_TODBERT_sampler_' + \
            f'{sampler_config}_drop_labels_{drop_config}.pkl'

    else:
        training_path = root + f'training/train_{embedding}_sampler_' + \
            f'{sampler_config}_drop_labels_{drop_config}.pkl'
        testing_path = root + f'testing/test_{embedding}_sampler_' + \
            f'{sampler_config}_drop_labels_{drop_config}.pkl'

    return training_path, testing_path


def build_dataloaders(config):

    for embedding in config.get('embedding_batch'):
        for sampler_config in config.get('sampler_batch'):
            for drop_config in config.get('drop_col_batch'):
                print(
                    f'Embedding: {embedding} |'
                    f'Sampler: {sampler_config} |'
                    f'Drop Social Pressure: {drop_config}\n'
                        )

                train_df, test_df = import_data(config)
                training_dataloader, val_loader = gen_loaders(
                    train_df, test_df, embedding,
                    config.get('max_token_len'), sampler_config)

                output_dataloaders(
                    config, training_dataloader,
                    val_loader, embedding, sampler_config, drop_config
                    )


def output_dataloaders(
    config, training_dataloader, val_loader,
    embedding, sampler_config, drop_config
        ):

    training_path, testing_path = gen_dloader_paths(
        config, embedding, sampler_config, drop_config)

    if val_loader:
        logging.info(f'Writing testing dataloader to: {testing_path}')
        with open(testing_path, 'wb') as f:
            pickle.dump(val_loader, f)

    with open(training_path, 'wb') as f:
        logging.info(f'Writing training dataloader to: {training_path}')
        pickle.dump(training_dataloader, f)


if __name__ == '__main__':

    args = getargs()
    config = load_config(args.config_path)
    build_dataloaders(config)
