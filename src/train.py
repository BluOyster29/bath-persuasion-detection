import sys
import torch
import os
import torch.nn as nn
from tqdm.auto import tqdm

module_dir = os.path.expanduser('~/repos/UoB/bath-persuasion-detection/models')
sys.path.append(module_dir)

from bert_classifier import BertClassifier, RNN # noqa
from utils import gen_metadata # noqa


def init_hyperparameeters(config, label_columns, vocab_size=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = config.get('num_epochs')
    criterion = nn.BCEWithLogitsLoss()

    if config.get('use_pretrained'):
        model = BertClassifier(config.get('pretrained_model').split('-')[0],
                               config.get('pretrained_model'),
                               len(label_columns))
    else:
        model = RNN(vocab_size, config, len(label_columns))

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=2e-5, weight_decay=1e-5)
    return device, num_epochs, criterion, model, optimizer


def train_model(
        model, training_dataloader, num_epochs, device, optimizer, criterion):

    avg_loss = 0

    model.train()

    with tqdm(range(num_epochs), desc='Average Epoch Loss: ') as t:
        for e in range(num_epochs):
            epoch_loss = []

            with tqdm(range(len(training_dataloader)), desc='Loss: 0') as t2:
                for b, batch in enumerate(training_dataloader):

                    optimizer.zero_grad()

                    if model.name == 'rnn':
                        input_ids = batch['input_ids'].to(device)
                        labels = batch['labels'].to(device)
                        embeddings = model.embd(input_ids)
                        outputs = model(embeddings)

                    elif model.name in ['bert', 'distilbert']:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        outputs = model(input_ids, attention_mask)

                    loss = criterion(outputs, labels.float())
                    epoch_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()

                    avg_epoch_loss = round(sum(epoch_loss)/len(epoch_loss), 4)
                    description = f'Epoch: {e} | '
                    description += f'Batch {b} | '
                    description += f'Average Loss: {avg_epoch_loss}'

                    t2.set_description(description)
                    t2.update()

            t.set_description(
                f'Average Epoch Loss: \
                    {round(sum(epoch_loss)/len(epoch_loss), 4)}')
            t.update()

    return avg_loss, model


def train(config, training_dataloader):
    """
    Trains a model using the provided configuration, label columns, and
    training dataloader.

    Args:
        config (dict): Configuration parameters for training.
        label_columns (list): List of column names for the labels.
        training_dataloader (DataLoader): DataLoader object containing
        the training data.

    Returns:
        tuple: A tuple containing the average loss and the trained model.
    """

    vocab_size = None
    label_columns = training_dataloader.dataset.label_columns
    if not config.get('use_pretrained'):
        vocab_size = training_dataloader.dataset.vocab_size
    device, num_epochs, criterion, model, optimizer = init_hyperparameeters(
        config, label_columns, vocab_size)
    avg_loss, trained_model = train_model(
        model, training_dataloader, num_epochs, device, optimizer, criterion)
    meta_data = gen_metadata(config, 'model')

    if config.get('output_model'):
        torch.save(trained_model.state_dict(),
                   f"{config.get('output_model')}_{meta_data}_{avg_loss}.pth")

    return trained_model
