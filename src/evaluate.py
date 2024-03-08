from tqdm.auto import tqdm
from utils import load_dataloader, load_model, load_config, get_args
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, roc_curve
from sklearn.metrics import f1_score, recall_score, precision_score, auc
import torch


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
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels.extend(predicted_probs.cpu().numpy() > 0.4)
            true_labels.extend(labels.cpu().numpy())

    return true_labels, predicted_labels


def gen_stats(true_labels, predicted_labels):

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels, predicted_labels, average='weighted', zero_division=True)
    recall = recall_score(
        true_labels, predicted_labels, average='weighted', zero_division=True)
    f1 = f1_score(
        true_labels, predicted_labels, average='weighted', zero_division=True)
    report = classification_report(true_labels, predicted_labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall, 'f1': f1,
        'report': report
    }


def roc_step_1(
    true_labels: list,
    predicted_labels: list,
    label_columns: list
) -> tuple:

    """
    Calculate the Receiver Operating Characteristic (ROC) curve for each class.

    Args:
        true_labels (list): A list of true labels.
        predicted_labels (list): A list of predicted labels.
        label_columns (list): A list of label columns.

    Returns:
        tuple: A tuple containing the false positive rate (fpr),
        true positive rate (tpr), and area under the ROC curve (roc_auc)
        for each class.
    """

    predicted_labels = np.array(predicted_labels).astype(int)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    true_labels = np.array(true_labels).astype(int)
    for i in range(len(label_columns)):  # num_classes is the number of classes
        fpr[i], tpr[i], _ = roc_curve(
            true_labels[:, i], predicted_labels[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc, fpr_micr, tpr_micr, roc_auc_micr,
                   label_columns, config):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        fpr (list): List of false positive rates for each class.
        tpr (list): List of true positive rates for each class.
        roc_auc (list): List of area under the ROC curve for each class.
        fpr_micr (list): List of false positive rates for micro-average.
        tpr_micr (list): List of true positive rates for micro-average.
        roc_auc_micr (float): Area under the ROC curve for micro-average.
        label_columns (list): List of class labels.
        config (dict): Configuration settings.

    Returns:
        None
    """

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr_micr, tpr_micr, label='Micro-average ROC curve (area = {0:0.2f})'
        ''.format(roc_auc_micr), color='deeppink', linestyle=':', linewidth=4)

    for i in range(len(label_columns)):
        plt.plot(
            fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
            ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)  # Plot diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    if config.get('show_curve'):
        plt.show()
    plt.savefig(config.get('roc_plot_path'))


def gen_roc_auc(true_labels, predicted_labels, label_columns, config):

    """
    Generate ROC AUC (Area Under the Curve) for binary classification.

    Args:
        true_labels (numpy.ndarray): True labels of the samples.
        predicted_labels (numpy.ndarray): Predicted labels of the samples.
        label_columns (list): List of column names for the labels.
        config (dict): Configuration settings.

    Returns:
        None
    """
    fpr, tpr, roc_auc = roc_step_1(
        true_labels, predicted_labels, label_columns)

    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(
        true_labels.ravel(), predicted_labels.ravel())

    roc_auc_micro = auc(fpr_micro, tpr_micro)

    plot_roc_curve(
        fpr, tpr, roc_auc, fpr_micro, tpr_micro,
        roc_auc_micro, label_columns, config
        )


def output_stats(stats, config):

    """
    Writes the evaluation statistics to a file and prints them to the console.

    Args:
        stats (dict): A dictionary containing the evaluation statistics.
        config (dict): A dictionary containing the configuration settings.

    Returns:
        None
    """

    with open(config.get('stats_path'), 'w') as f:
        f.write(f"Accuracy: {stats['accuracy']}\n")
        f.write(f"Precision: {stats['precision']}\n")
        f.write(f"Recall: {stats['recall']}\n")
        f.write(f"F1: {stats['f1']}\n")
        f.write(f"Classification Report: {stats['report']}\n")

    print(f"Accuracy: {stats['accuracy']}")
    print(f"Precision: {stats['precision']}")
    print(f"Recall: {stats['recall']}")
    print(f"F1: {stats['f1']}")
    print(f"Classification Report: {stats['report']}")


def evaluate(config, eval_loader=None, trained_model=None):

    label_columns = eval_loader.dataset.num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.get('From saved mode'):
        trained_model = load_model(config.get('model_path'), label_columns)

    if not eval_loader:
        eval_loader = load_dataloader(config)

    true_labels, predicted_labels = eval_model(
        trained_model, eval_loader, device)

    stats = gen_stats(true_labels, predicted_labels)
    gen_roc_auc(true_labels, predicted_labels, label_columns, config)
    output_stats(stats, config)


if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config_path)
    evaluate(config)
