import pandas as pd
import torch
import pickle as pkl
from datetime import datetime

from torch.utils.data import DataLoader
from collections import Counter

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import WeightedRandomSampler
from persuasionDataset import PersuasionDataset
from utils import load_config, getargs
import warnings

# Ignore specific warning types
warnings.filterwarnings("ignore", category=FutureWarning)

def output(ob, path):
    path = path + datetime.now().strftime("%Y%m%d%H%M%S") + '.pkl'
    with open(path, 'wb') as file:
        pkl.dump(ob, file)


def load_data(config: dict):

    df = pd.concat([
        pd.read_csv(f"{config.get('data_root_path')}training.csv"),
        pd.read_csv(f"{config.get('data_root_path')}testing.csv")]
    )

    train_df, eval_df = train_test_split(
        df, test_size=config.get('test_size')
    )

    if config.get('verbose'):
        print(f'Num Training {len(train_df)}')
        print(f'Num Evaluation {len(eval_df)}')

    return train_df, eval_df


def gen_sample_weights(training_dataset):
    labels = [torch.argmax(label['labels']).item() for label in
              training_dataset]  # Assuming your dataset returns (input_vector, label) pairs
    class_distribution = Counter(labels)

    # Calculate weights for each sample
    class_weights = {class_label: len(training_dataset) / (len(class_distribution) * class_count) for
                     class_label, class_count in class_distribution.items()}
    weights = [class_weights[label] for label in labels]

    # Convert weights to tensor
    weights_tensor = torch.tensor(weights, dtype=torch.float)

    # Create a sampler to balance the dataset during training
    sampler = WeightedRandomSampler(weights_tensor, len(weights_tensor))
    return sampler


def load_dataframes(config: dict, concat=None):
    train_df, eval_df = load_data(config)
    return pd.concat([train_df, eval_df])


def load_dataset(df:pd.DataFrame, config: dict):

    dataset = PersuasionDataset(
        df,
        config.get('model')[config.get('model_setup')]['embeddings'],
        config
    )

    return dataset


def gen_dataloader(df, config, train=None):
    """
    docstring for gen_dataloader

    :param df:
    :param config:
    :param train:

    """

    print('Loading data')
    dataset = load_dataset(df, config)

    if config.get('return_dataset') and train:
        output(dataset, config.get('training_dataset_path'))
    elif config.get('return_dataset'):
        output(dataset, config.get('testing_dataset_path'))

    if train:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.get('hyperparameters').get('batch_size'),
            sampler=gen_sample_weights(dataset)
        )

    else:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True
        )


def preprocess(config: dict):

    df = load_dataframes(config, concat=True)
    train_df, eval_df = train_test_split(df, test_size=config.get('test_size'))
    training_dataloader = gen_dataloader(train_df, config, train=True)
    testing_dataloader = gen_dataloader(eval_df, config)
    return training_dataloader, testing_dataloader


if __name__ == '__main__':
    args = getargs()
    config_ob = load_config(args.config_path)
    preprocess(config_ob)
