from .models import *

__all__ = ["create_models"]

MODELS = {
    "fedavg_cifar": fedavgnet.FedAvgNetCIFAR,
    "medmnist": fedavgnet.MedMNISTNet
}

NUM_CLASSES = {
    "cifar10": 10,
    "BloodMNIST": 8,
    "OrganCMNIST": 11,
    "OrganSMNIST": 11
}


def create_models(model_name, dataset_name):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]

    if dataset_name == 'OrganCMNIST':
        ch = 1
    elif dataset_name == 'OrganSMNIST':
        ch = 1
    else:
        ch = 3

    model = MODELS[model_name](num_classes=num_classes, in_channels=ch)

    return model
