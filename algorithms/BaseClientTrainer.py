import numpy as np
from .measures import model_metrics
import torch
import torch.nn as nn
import copy
from .measures import *

__all__ = ["BaseClientTrainer"]



class BaseClientTrainer:
    def __init__(self, algo_params, model, local_epochs, device, num_classes, save_folder,
                 dataset='olives', temp_trainloader=None):
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        # Params
        self.local_epochs = local_epochs
        self.device = device
        self.datasize = None
        self.num_classes = num_classes

        # algorithm-specific parameters
        self.algo_params = algo_params
        self.round = None

        # model & optimizer
        self.model = model
        self.lr = 0
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr)

        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()
        self.multilabel = False

        self.trainloader = temp_trainloader
        self.global_testloader = None
        self.local_testloader = None
        self.current_client = None
        self.testsize = None
        self.global_test_size = 0

        self.prev_acc = {}
        self.stats = {}

    def train(self):
        """Local training"""
        local_size = self.datasize
        # Train local model
        self.model.train()
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        for ep in range(self.local_epochs):
            for data, targets, _ in self.trainloader:
                self.optimizer.zero_grad()
                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, targets)
                # backward pass
                loss.backward()
                self.optimizer.step()

        # used trained model to get local TRAIN accuracy and TEST accuracy
        local_results = self._get_local_stats(current_client=self.current_client)
        return local_results, local_size

    def _get_local_stats(self, current_client):
        # This typically evaluates local models on global test sets
        local_results = {}
        # local results train accuracy
        local_results["train_acc"] = evaluate_model(
            self.model, self.trainloader, self.dataset, self.device
        )
        # Set up forgetting counters
        # Local test sets
        try:
            previous_client_acc = self.prev_acc[current_client]
        except KeyError:
            previous_client_acc = np.zeros(shape=self.testsize)

        # local test accuracy calculation

        new, forgets_local, nfr_local, local_acc = model_metrics(self.model, self.local_testloader,
                                                                 previous_acc=previous_client_acc,
                                                                 dataset=self.dataset)

        _, loc_on_loc_classwise, ll = evaluate_model_classwise(model=self.model, dataloader=self.local_testloader,
                                                           num_classes=self.num_classes, device=self.device)
        ll = np.mean(ll)
        loc_acc = local_acc

        # update saved accuracy for client (LOCAL MODELS --> LOCAL TEST SET)
        self.prev_acc[current_client] = new.astype(int)
        # Convert results to DF
        local_results['Sampled Client'] = current_client
        local_results["local on local nfr"] = nfr_local
        local_results["local on local test acc"] = loc_acc
        local_results["local on local classwise"] = loc_on_loc_classwise
        local_results["local on local classwise tot"] = ll

        return local_results

    def download_global(self, server_weights, server_optimizer):
        """Load model & Optimizer"""
        self.model.load_state_dict(server_weights)
        self.optimizer.load_state_dict(server_optimizer)

    def upload_local(self):
        """Uploads local model's parameters"""
        local_weights = copy.deepcopy(self.model.state_dict())

        return local_weights

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.trainloader = None
        self.global_testloader = None
        self.local_testloader = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        self.current_client = None
        self.testsize = None
        self.global_test_size = 0

    def _keep_global(self):
        """Keep distributed global model's weight"""
        self.dg_model = copy.deepcopy(self.model)
        self.dg_model.to(self.device)

        for params in self.dg_model.parameters():
            params.requires_grad = False
