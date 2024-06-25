from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score


def gen_stats(true_labels, predicted_labels, verbose=None):

    accuracy = accuracy_score(
        true_labels,
        predicted_labels)

    precision = precision_score(true_labels, predicted_labels,
                                zero_division=True)
    recall = recall_score(true_labels, predicted_labels,
                          zero_division=True)
    f1 = f1_score(true_labels, predicted_labels,
                  zero_division=True)

    if verbose:
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

    return accuracy, precision, recall, f1


def gen_classification_report(true_labels, predicted_labels, verbose=None):

    report = classification_report(
        true_labels, predicted_labels, zero_division=True)

    if verbose:
        print(report)

    return report
