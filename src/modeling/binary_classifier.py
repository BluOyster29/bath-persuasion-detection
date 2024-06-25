import torch.nn as nn
from transformers import RobertaForSequenceClassification

# Load pre-trained BERT model and tokenizer


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

    def __init__(self, name, bert_model, num_labels):
        super(BertClassifier, self).__init__()
        self.name = name
        self.bert = RobertaForSequenceClassification.from_pretrained(
            bert_model)

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
        if self.name in ['bert', 'roberta']:
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
            pooled_output = outputs.pooler_output

        elif self.name == 'distilbert':
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]

        else:
            raise ValueError("Invalid model name")

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
