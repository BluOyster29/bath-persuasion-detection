from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


class PersuasionStrategyDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_token_len
            ):

        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.LABEL_COLUMNS = data.columns.tolist()[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        data_row = self.data.iloc[index]
        comment_text = data_row.text
        labels = data_row[self.LABEL_COLUMNS]
        encoding = self.tokenizer.encode_plus(
            comment_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            )

        return dict(
            comment_text=comment_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )


class PersuasionStrategyDatasetLSTM(Dataset):
    def __init__(self, data: pd.DataFrame, vocab):
        self.encoded_text = self.df_to_tensor(data.encoded.tolist())
        self.labels = data.iloc[:, 1:-1].values.tolist()
        self.LABEL_COLUMNS = data.columns.tolist()[1:-1]
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def df_to_tensor(self, data):
        tensors = []
        for row in data:
            row = np.array([int(i) for i in row.split()])
            tensors.append(torch.LongTensor(row))
        return pad_sequence(tensors, batch_first=True, padding_value=0)

    def __len__(self):
        return len(self.encoded_text)

    def __getitem__(self, index: int):
        text = self.encoded_text[index]
        labels = self.labels[index]
        return dict(
            input_ids=text,
            labels=torch.LongTensor(labels)
        )
