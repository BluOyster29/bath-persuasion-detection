import pandas as pd
import os
from sklearn.model_selection import train_test_split
from utils.utils import getargs, load_config, find_pers_type
from torch.utils.data import DataLoader
from utils.data import gen_sampler
import pickle
import warnings
import logging

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings('ignore', category=FutureWarning)


def fetch_data(path_to_training, path_to_testing=None):

    if not path_to_testing:
        df = pd.read_csv(path_to_training)

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


def gen_datasets(df, label, tokenizer, task):

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True,
    )

    if task == 'multilabel':
        from persuasion_detection import PersuasionStrategyMultilabelDataset \
            as PersuasionDataset

    elif task == 'binary_classification':
        from persuasion_detection import PersuasionStrategyDataset \
            as PersuasionDataset

    train_ds = PersuasionDataset(
        train_df, label, tokenizer
    )

    test_ds = PersuasionDataset(
        test_df, label, tokenizer
        )

    return train_ds, test_ds


def gen_dataloaders(dataset, batch_size, test=None):

    if test:
        sampler = None
    else:
        sampler = gen_sampler(dataset)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size
    )
    return dataloader


def gen_dloader_paths(config, embedding, sampler_config, drop_config):

    root = config.get('output_data_path')

    if config.get('task') == 'binary_classification':
        pers_strat = find_pers_type(config.get('path_to_training'))

    if not os.path.exists(root+config.get('task')):
        os.makedirs(root + config.get('task') + '/')

    root += config.get('task') + '/'

    if config.get('task') == 'binary_classification' and os.path.exists(root + pers_strat) == False:
        os.makedirs(root + pers_strat + '/')
        root = root + pers_strat + '/'
        os.makedirs(root+'training/')
        os.makedirs(root+'testing/')

    if embedding == 'TODBERT/TOD-BERT-JNT-V1':
        prefix = 'TODBERT'
    else:
        prefix = embedding

    training_path = (
        f"{root}training/train_{prefix}_sampler_{sampler_config}_"
        f"drop_labels_{drop_config}_{config.get('task')}.pkl"
    )
    testing_path = (
        f"{root}testing/test_{prefix}_sampler_{sampler_config}_"
        f"drop_labels_{drop_config}_{config.get('task')}.pkl"
    )

    return training_path, testing_path


def build_dataloaders(config):

    if config.get('task') == 'binary_classification':
        config['persuasion_type'] = find_pers_type(
            config.get('path_to_training'))

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
                    train_df, test_df, embedding, config.get('task'),
                    config.get('max_token_len'), sampler_config)

                output_dataloaders(
                    config, training_dataloader, val_loader,
                    embedding, sampler_config, drop_config
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
