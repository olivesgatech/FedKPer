import numpy as np
import torch
import os
import sys
import copy
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from algorithms.BaseClientTrainer import BaseClientTrainer

__all__ = ["ClientTrainer"]


class ClientTrainer(BaseClientTrainer):
    def __init__(self, criterion, **kwargs):
        super(ClientTrainer, self).__init__(**kwargs)
        """
        ClientTrainer class contains local data and local-specific information.
        After local training, upload weights to the Server.
        """
        self.criterion = criterion
        self.classifier = None
        self.prior_stats = None
        self.CE = nn.CrossEntropyLoss()
        self.counts = None
        self.class_distribution = None

    def train(self):
        """Local training"""

        self.counts = np.zeros(self.num_classes)
        for cls, n in self.class_distribution.items():
            self.counts[int(cls)] = float(n)

        # Keep global model's weights
        self._keep_global()

        self.model.train()
        self.model.to(self.device)

        local_size = self.datasize

        kl_items = []
        beta_items = []
        ce_items = []

        for _ in range(self.local_epochs):
            for data, targets, _ in self.trainloader:
                self.optimizer.zero_grad()

                # forward pass
                data, targets = data.to(self.device), targets.to(self.device)
                logits, dg_logits = self.model(data), self._get_dg_logits(data)

                loss, kl, beta, ce = self.criterion(logits, targets, dg_logits)
                kl_items.append(kl)
                beta_items.append(beta)
                ce_items.append(ce)

                # backward pass
                loss.backward()
                self.optimizer.step()

        local_results = self._get_local_stats(current_client=self.current_client)
        score = self.diversity_aware_weights(local_results['train_acc'])

        return local_results, local_size, score

    def normalized_entropy_from_counts(self, counts, eps: float = 1e-12):

        counts = counts.astype(np.float64)
        s = counts.sum()
        if s <= 0:
            return 0.0
        p = counts / (s + eps)
        H = -(p * np.log(p + eps)).sum()  # entropy
        Hmax = np.log(len(counts) + eps)  # max entropy for C classes
        val = float(H / (Hmax + eps))
        return float(np.clip(val, 0.0, 1.0))

    def diversity_aware_weights(self, acc, eps: float = 1e-6,):
        div = self.normalized_entropy_from_counts(self.counts, eps=1e-12)
        score = max(acc, 0.0) * (eps + div)
        return score

    def _get_dg_logits(self, data):

        with torch.no_grad():
            dg_logits = self.dg_model(data)

        return dg_logits

    def _prior_local_logits(self, data):

        with torch.no_grad():
            logits = self.prior_local(data)

        return logits

    def upload_local_classifier(self):
        """Uploads local model's classifier"""
        local_classifier = copy.deepcopy(self.model)

        return local_classifier

    def _old_local(self):
        self.prior_local = copy.deepcopy(self.classifier)
        self.prior_local.to(self.device)

        for params in self.prior_local.parameters():
            params.requires_grad = False
