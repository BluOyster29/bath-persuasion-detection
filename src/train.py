import sys
sys.path.append('/Users/rt853/repos/UoB/bath-persuasion-detection/models')

from tqdm.auto import tqdm
from bert_classifier import BertClassifier
import torch.nn as nn 
import torch

from utils import gen_metadata

def init_hyperparameeters(config, label_columns):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = config.get('num_epochs')
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for multilabel classification

    model = BertClassifier(config.get('pretrained_model'), len(label_columns))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    return device, num_epochs, criterion, model, optimizer

def train_model(model, training_dataloader, num_epochs, device, optimizer, criterion):

    avg_loss = 0
    model.train()

    with tqdm(range(num_epochs), desc='Average Epoch Loss: ') as t:
        for _ in range(num_epochs):
            epoch_loss = []
            
            with tqdm(range(len(training_dataloader)), desc='Loss: 0') as t2:
                for _, batch in enumerate(training_dataloader):

                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    optimizer.zero_grad()

                    outputs = model(input_ids, attention_mask)

                    loss = criterion(outputs, labels.float())
                    epoch_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()

                    t2.set_description(
                        f'Loss: {round(sum(epoch_loss)/len(epoch_loss),4)}')
                    t2.update()

        t.set_description(
        f'Average Epoch Loss: {round(sum(epoch_loss)/len(epoch_loss),4)}')
        t.update()

    return avg_loss, model

def train(config, label_columns, training_dataloader):
    """
    Trains a model using the provided configuration, label columns, and training dataloader.

    Args:
        config (dict): Configuration parameters for training.
        label_columns (list): List of column names for the labels.
        training_dataloader (DataLoader): DataLoader object containing the training data.

    Returns:
        tuple: A tuple containing the average loss and the trained model.
    """
    device, num_epochs, criterion, model, optimizer = init_hyperparameeters(config, label_columns)
    avg_loss, trained_model = train_model(model, training_dataloader, num_epochs, device, optimizer, criterion)
    meta_data = gen_metadata(config, 'model')
    
    if config.get('output_model'): 
        torch.save(trained_model.state_dict(), f"{config.get('output_model')}_{meta_data}_{avg_loss}.pth")
        
    return trained_model
