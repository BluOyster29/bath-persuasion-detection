import torch
from tqdm.auto import tqdm


def eval_model(model, eval_loader, device):
    """
    Evaluate the performance of a model on the evaluation data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        eval_loader (torch.utils.data.DataLoader): The data loader for
        the evaluation data.
        device (torch.device): The device to run the evaluation on.

    Returns:
        tuple: A tuple containing the true labels and predicted labels.
    """

    true_labels = []
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_loader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)

            predicted_probs = torch.sigmoid(outputs.logits)
            predicted_labels.append(predicted_probs.argmax().item())
            # predicted_labels.extend(predicted_probs.cpu().numpy() > 0.7)
            true_labels.append(labels.argmax().item())
    return true_labels, predicted_labels
