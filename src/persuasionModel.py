import torch.nn as nn
from transformers import AutoModel


class TransformerMultilabelClassifier(nn.Module):
    
    def __init__(self, pretrained_model, num_labels):
        super(TransformerMultilabelClassifier, self).__init__()
        self.pretrained_model = pretrained_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
