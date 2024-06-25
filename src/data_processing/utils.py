import torch
from sklearn.model_selection import train_test_split
from .persuasion_strategy_dataset import binaryPersuasionDataset, gen_sampler
from torch.utils.data import DataLoader


def encode_label(label):

    empty_tens = torch.zeros(2)
    empty_tens[label] += 1
    return empty_tens


def gen_datasets(df, label, tokenizer, eval=None):

    if not eval:
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            shuffle=True,
        )

    else:
        train_df = df
        test_df = None

    train_ds = binaryPersuasionDataset(
        train_df,
        label,
        tokenizer
    )

    if eval:
        return train_ds, None

    test_ds = binaryPersuasionDataset(
        test_df,
        label,
        tokenizer
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
