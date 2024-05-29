from torch.utils.data import Dataset

import torch
import pandas as pd
from transformers import AutoTokenizer


class PersuasionStrategyBinaryDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: str,
        max_token_len
            ):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.data = data
        self.max_token_len = max_token_len
        self.label_columns = [0, 1]
        self.tokenizer_name = tokenizer
    
    def encode(self, text):

        if self.tokenizer_name == 'roberta':
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True, max_length=self.max_token_len,
                return_token_type_ids=False, padding="max_length",
                truncation=True, return_attention_mask=True,
                return_tensors='pt',
                )

        else:

            encoding = self.tokenizer(
                text, truncation=True, max_length=self.max_length,
                padding='max_length', return_tensors='pt'
            )

        return encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        data_row = self.data.iloc[index]
        text = data_row.text
        labels = data_row[self.LABEL_COLUMNS]
        encoding = self.encode(text)

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )


class PersuasionStrategyMultilabelDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: str,
        max_token_len
            ):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.data = data
        self.max_token_len = max_token_len
        self.label_columns = data.columns.tolist()[1:]
        self.tokenizer_name = tokenizer

    def encode(self, text):

        if self.tokenizer_name == 'roberta':
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True, max_length=self.max_token_len,
                return_token_type_ids=False, padding="max_length",
                truncation=True, return_attention_mask=True,
                return_tensors='pt',
                )

        else:

            encoding = self.tokenizer(
                text, truncation=True, max_length=self.max_length,
                padding='max_length', return_tensors='pt'
            )

        return encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        data_row = self.data.iloc[index]
        text = data_row.text
        labels = data_row[self.LABEL_COLUMNS]
        encoding = self.encode(text)

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )
