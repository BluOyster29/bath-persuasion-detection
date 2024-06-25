from tqdm.auto import tqdm


def forward_pass(model_name, batch, model, input_ids, attention_mask, labels):

    if model_name in ['bert', 'distilbert']:
        outputs = model(input_ids, attention_mask)

    elif model_name == 'roberta':
        outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)

    return outputs, labels


def update_pb(epoch_loss, epoch, batch):
    avg_epoch_loss = round(sum(epoch_loss)/len(epoch_loss), 4)
    description = f'Epoch: {epoch} | '
    description += f'Batch {batch} | '
    description += f'Average Loss: {avg_epoch_loss}'
    return description


def train_model(model, training_dataloader, num_epochs,
                device, optimizer, criterion, model_name):

    avg_loss = 0

    model.train()

    with tqdm(range(num_epochs), desc='Average Epoch Loss: ') as pbar1:
        for e in range(num_epochs):
            epoch_loss = []

            with tqdm(range(len(training_dataloader)),
                      desc='Loss: 0') as pbar2:

                for b, batch in enumerate(training_dataloader):

                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    optimizer.zero_grad()

                    outputs, labels = forward_pass(
                        'roberta', batch, model, input_ids,
                        attention_mask, labels)

                    if model_name == 'roberta':
                        loss = outputs.loss

                    else:
                        loss = criterion(outputs, labels.float())

                    epoch_loss.append(loss.item())
                    loss.backward()
                    optimizer.step()

                    description = update_pb(epoch_loss, e, b)
                    pbar2.set_description(description)
                    pbar2.update()

            pbar1.set_description(
                'Average Epoch Loss: ' +
                f'{round(sum(epoch_loss)/len(epoch_loss), 4)}')

            pbar1.update()

    return avg_loss, model
