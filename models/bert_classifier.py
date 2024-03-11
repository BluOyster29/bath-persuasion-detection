import torch.nn as nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """
    BERT-based classifier model.

    Args:
        bert_model (str): The pre-trained BERT model to use.
        num_labels (int): The number of output labels.

    Attributes:
        bert (BertModel): The BERT model.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Linear): Linear layer for classification.

    """

    def __init__(self, bert_model, num_labels):
        super(BertClassifier, self).__init__()
        self.name = 'bert'
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the BERT classifier.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.

        Returns:
            torch.Tensor: The logits for each class.

        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class RNN(nn.Module):
    def __init__(self, vocab_size, config, output_size):
        super(RNN, self).__init__()
        self.name = 'rnn'
        self.embedding_size = config.get('embedding_size')
        self.hidden_size = config.get('hidden_size')
        self.embd = nn.Embedding(
            vocab_size, self.embedding_size, padding_idx=0)
        self.rnn = nn.GRU(
            self.embedding_size, self.hidden_size, batch_first=True,
            num_layers=2, dropout=0.1, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.hidden_size*2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.embd(x)  # Convert word indices to word embeddings
        out, _ = self.rnn(x)
        out = self.dropout(self.relu(self.fc(out[:, -1, :])))
        return out


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
