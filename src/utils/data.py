from torch.utils.data import WeightedRandomSampler
import torch


def gen_sampler(dataset, label_name):
    """
    Generate a weighted random sampler based on the class distribution of
    the dataset.

    Args:
        dataset (Dataset): The dataset containing the data.
        label_name (str): The name of the label column in the dataset.

    Returns:
        WeightedRandomSampler: The weighted random sampler.
    """

    class_counts = dataset.data[label_name].value_counts().to_list()
    num_samples = sum(class_counts)
    labels = dataset.data.binary_label.tolist()
    class_weights = [
        num_samples/class_counts[i] for i in range(len(class_counts))]

    weights = [
        class_weights[labels[i]] for i in range(int(num_samples))]

    sampler = WeightedRandomSampler(
        torch.DoubleTensor(weights), int(num_samples)
        )

    return sampler
