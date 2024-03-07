from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch


class PersuasionDataset(Dataset):

    def __init__(self, data, tokenizer, config):

        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer
        )
        self.data = data
        self.max_token_len = config.get("max_token_length")
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
