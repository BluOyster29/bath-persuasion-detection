import torch.nn as nn
import torch
import time
from tqdm.auto import tqdm

from persuasionModel import TransformerMultilabelClassifier
from transformers import BertModel
from preprocess import preprocess
from utils import load_config, getargs


def load_params(config):

    model_embeddings = config.get('model')[config.get('model_setup')].get('embeddings')
    model_desc = config.get('model')[config.get('model_setup')].get('name')
    device = config.get('hyperparameters').get('device')
    pretrained_model = BertModel.from_pretrained(model_embeddings)
    return pretrained_model, model_desc, device


def training_setup(config, pretrained_model, device, num_labels):

    model = TransformerMultilabelClassifier(pretrained_model, num_labels)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.get('hyperparameters').get('learning_rate'))
    model.to(device)
    return model, criterion, optimizer


def train(config):

    pretrained_model, model_desc, device = load_params(config)

    model, criterion, optimizer = training_setup(
        config, pretrained_model, device, len(config.get('labels'))
    )

    epoch_losses = []
    training_dataloader, testing_dataloader = preprocess(config)
    num_epochs = config.get('hyperparameters').get('num_epochs')
    with tqdm(range(num_epochs), desc='Epoch: ') as pbar1:
        for epoch_num in range(num_epochs):
            pbar1.set_description(f'Epoch: {epoch_num+1}')
            epoch_loss = train_epoch(training_dataloader, model, criterion, optimizer, device)
            pbar1.update(1)
            epoch_losses.append(epoch_loss)

    return model, epoch_losses


def train_epoch(training_dataloader, model, criterion, optimizer, device):

    epoch_loss = 0
    with tqdm(training_dataloader, desc='Training: ') as pbar2:
        for batch_idx, batch in enumerate(training_dataloader):
            pbar2.update(1)
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (batch_idx + 1) % (len(training_dataloader) // 5) == 0:
                pbar2.set_description(f'Avg Epoch Loss: {round(epoch_loss / len(training_dataloader), 8)}')

    return epoch_loss/len(training_dataloader)


if __name__ == '__main__':
    args = getargs()
    cnfg = load_config(args.config_path)
    trained_model = train(cnfg)