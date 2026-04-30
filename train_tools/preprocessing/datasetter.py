import glob
import numpy as np
import os
from .medmnist.loader import get_dataloader_medmnist, get_all_targets_medmnist
from .cifar10.loader import get_all_targets_cifar10, get_dataloader_cifar10
from collections import Counter

__all__ = ["data_distributer"]

DATA_INSTANCES = {
    "cifar10": get_all_targets_cifar10,
    "BloodMNIST": get_all_targets_medmnist,
    "OrganCMNIST": get_all_targets_medmnist,
    "OrganSMNIST": get_all_targets_medmnist,
}

DATA_LOADERS = {
    "cifar10": get_dataloader_cifar10,
    "BloodMNIST": get_dataloader_medmnist,
    "OrganCMNIST": get_dataloader_medmnist,
    "OrganSMNIST": get_dataloader_medmnist
}


def data_distributer(
    args,
    root,
    dataset_name,
    batch_size,
    n_clients,
    partition,
    save_folder
):
    """
    Distribute dataloaders for server and locals by the given partition method.
    """
    root = os.path.join(root, dataset_name)
    # Get all available classes for train samples
    all_targets = DATA_INSTANCES[dataset_name](root, dataset_label=dataset_name)
    # Figure out number of classes
    num_classes = len(np.unique(all_targets))
    print('Class count: ', num_classes)

    net_dataidx_map_test = None

    local_loaders = {
        i: {"datasize": 0, "train": None, "test": None, "test_size": 0, "class_distribution": {}} for i in range(n_clients)
    }


    contents = glob.glob(save_folder + '*')

    try:
        net_dataidx_map = np.load(save_folder + 'train_idxs.npy', allow_pickle=True).item(0)
        net_dataidx_map_test = np.load(save_folder + 'test_idxs.npy', allow_pickle=True).item(0)
        print('loaded')
    except FileNotFoundError:
        net_dataidx_map = partition_class_samples_with_dirichlet_distribution(alpha=partition.alpha,
                                                                              client_num=n_clients,
                                                                              targets=all_targets,
                                                                              class_num=num_classes)
        # print(net_dataidx_map)
        net_dataidx_map_test, net_dataidx_map = create_local(idxs=net_dataidx_map, all_targets=all_targets,
                                                             save_folder=save_folder)

    print(">>> Distributing client train data...")
    print(save_folder)
    for client_idx, dataidxs in net_dataidx_map.items():
        local_loaders[client_idx]["train"] = DATA_LOADERS[dataset_name](
            root, mode='tr', batch_size=batch_size, dataidxs=dataidxs, dataset_label=dataset_name
        )
        local_loaders[client_idx]["datasize"] = len(dataidxs)
        cur_classes = all_targets[dataidxs]
        local_classes = dict(Counter(cur_classes))
        local_loaders[client_idx]["class_distribution"] = local_classes

    if net_dataidx_map_test is not None:
        print(">>> Distributing client test data...")
        # this is for local test set; local test set is derived ultimately from 'train' split
        # global test set is derived from 'test' split
        m = 'tr'

        for client_idx, dataidxs in net_dataidx_map_test.items():
            # Note: Must train mode if not wanting to use train set (here: local client test set is made from local client data)
            local_testloader = DATA_LOADERS[dataset_name](
                root, mode=m, batch_size=batch_size, dataidxs=dataidxs, dataset_label=dataset_name
            )

            local_loaders[client_idx]["test"] = local_testloader
            local_loaders[client_idx]["test_size"] = len(dataidxs)

    ################################################################################################################
    # Global Dataloader (For testing generalization)
    test_global_loader = DATA_LOADERS[dataset_name](root, mode='te', batch_size=batch_size, dataset_label=dataset_name)
    global_loaders = {
        "test": test_global_loader,
        "test_size": int(len(test_global_loader) * batch_size)
    }
    ###############################################################################################################

    data_distributed = {
        "global": global_loaders,
        "local": local_loaders,
        "num_classes": num_classes,
    }

    return data_distributed


def create_local(idxs, all_targets, amount=0.20, save_folder='/path/'):
    # Creating local test clients from partitioned data
    n_clients = len(idxs)
    net_dataidx_test = {i: np.array([], dtype="int64") for i in range(n_clients)}
    for i in range(len(idxs)):
        current_client_idxs = idxs[i]
        local_test_amount = int(len(current_client_idxs)*amount)
        # get unique classes
        classes = all_targets[current_client_idxs]
        unique_classes = np.unique(classes)
        num_classes = len(unique_classes)
        per_class = int(local_test_amount/num_classes)
        # for client's total num of classes, select local test idxs
        test_idxs = np.array([])
        for c in range(len(unique_classes)):
            curr_class = unique_classes[c]
            class_idxs = current_client_idxs[np.where(all_targets[current_client_idxs] == curr_class)]
            try:
                test_idxs_class = np.random.choice(class_idxs, size=per_class, replace=False)
            except ValueError:
                test_idxs_class = np.random.choice(class_idxs, size=int(amount*len(class_idxs)), replace=False)
            test_idxs = np.concatenate((test_idxs, test_idxs_class))
        # if test idxs are still empty, randomly select samples
        if len(test_idxs) == 0:
            test_idxs = np.random.choice(current_client_idxs, size=int(amount*len(current_client_idxs)), replace=False)
        # For each client, get idxs corresponding to local test set
        net_dataidx_test[i] = test_idxs.astype(int)
        new_train = np.where(np.isin(current_client_idxs, test_idxs, invert=True))[0]
        idxs[i] = current_client_idxs[new_train].astype(int)

    np.save(save_folder + 'test_idxs.npy', net_dataidx_test, allow_pickle=True)
    np.save(save_folder + 'train_idxs.npy', idxs, allow_pickle=True)
    return net_dataidx_test, idxs


def dirichlet(N, alpha, client_num, idx_batch, idx_k):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size

def partition_class_samples_with_dirichlet_distribution(
    alpha, client_num, targets, class_num
):
    net_dataidx_map = {}
    min_size = 0
    min_require_size = 10
    N = len(targets)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(client_num)]
        for k in np.unique(targets):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, client_num))

            # get the index in idx_k according to the dirichlet distribution
            proportions = np.array(
                [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            # generate the batch list for each client
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(len(idx_batch)):
        idx_batch[j] = np.array(idx_batch[j]).astype('int')

    for j in range(client_num):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

