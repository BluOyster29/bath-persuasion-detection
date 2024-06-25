from torch.utils.data import Dataset
from utils.utils import encode_label
import torch
from torch.utils.data import WeightedRandomSampler
import pandas as pd
from transformers import AutoTokenizer


def gen_sampler(dataset,):

    class_counts = dataset.data.binary_label.value_counts().to_list()
    num_samples = sum(class_counts)
    labels = dataset.data.binary_label.tolist()
    class_weights = [
        num_samples/class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(weights), int(num_samples))
    return sampler


class binaryPersuasionDataset(Dataset):
    def __init__(
            self, data, label, tokenizer
            ):

        self.data = data.sample(frac=1)
        self.label = label
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = 60

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        data_row = self.data.iloc[index]
        text = data_row.text
        binary_label = data_row.binary_label
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_length,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        return dict(
          text=text,
          input_ids=encoding["input_ids"].flatten(),
          attention_mask=encoding["attention_mask"].flatten(),
          labels=encode_label(binary_label)
        )


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

    def encode_text(self, text):

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
        labels = encode_label(data_row.binary_label)
        encoding = self.encode_text(text)

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
