import copy
import time
import os
import sys
import pandas as pd
import numpy as np
import torch
from timeit import default_timer as timer
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from algorithms.BaseServer import BaseServer
from algorithms.fedkper.ClientTrainer import ClientTrainer
from algorithms.fedkper.loss import *
from collections import Counter

__all__ = ["Server"]


class Server(BaseServer):
    def __init__(
        self, algo_params, model, data_distributed, optimizer, scheduler, dataset,
            save_folder, **kwargs
    ):
        super(Server, self).__init__(
            algo_params, model, data_distributed, optimizer, scheduler, dataset, save_folder, **kwargs
        )
        local_criterion = self._get_local_criterion()

        self.client = ClientTrainer(
            local_criterion,
            algo_params=self.algo_params,
            model=copy.deepcopy(model),
            local_epochs=self.local_epochs,
            device=self.device,
            num_classes=self.num_classes,
            save_folder=self.save_folder,
            dataset=self.dataset
        )

        self.client_classifiers = {}
        self.sigma = 1
        self.decay = 10

        print("\n>>> Title!! Server initialized...\n")

    def run(self):
        """Run the FL experiment"""
        self._print_start()

        #print(self.resume)

        for round_idx in range(self.n_rounds):

            print('Round ' + str(round_idx))
            self.round = round_idx

            start_time = timer()
            # Make local sets to distributed to clients
            sampled_clients = self._client_sampling(round_idx)
            # modify client history save by associating clients with round sampled
            self.client_history[round_idx] = sampled_clients
            print('LENGTH OF SAMPLED CLIENTS: ', len(sampled_clients))
            # Client training stage to upload weights & stats
            updated_local_weights, client_sizes, round_results, custom_weights = self._clients_training(
                sampled_clients, round_idx
            )
            print('for all clients, round results: ', round_results)
            #############################################################
            # Compute average accuracy and NFR
            round_info = self.local_clients_info[self.local_clients_info['Round'] == round_idx]
            accuracies = round_info['local on local classwise tot'].to_numpy()

            if np.isnan(np.sum(np.array(custom_weights))):
                ag_weights = self._aggregation(updated_local_weights, client_sizes)
            else:
                ag_weights = self._aggregation(updated_local_weights, custom_weights)

            # Update global weights and evaluate statistics
            self.df_glob = self._update_and_evaluate(ag_weights, round_results, round_idx, start_time,
                                                     sampled_clients=sampled_clients, df_global=self.df_glob)
            # End time
            end_time = timer()

            elapsed_time = end_time - start_time
            self.df_glob.at[round_idx, 'time cost'] = elapsed_time
            #self.df_glob.at[round_idx, 'flops'] = counts
            # Save temporary model
            model_path = os.path.join(self.save_folder, "model.pth")
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, model_path)
            # save spreadsheet
            self.df_glob.to_excel(self.save_folder + 'global_results.xlsx', index=False)
            self.df_local.to_csv(self.save_folder + 'global_on_local.csv', index=False)
            self.local_clients_info.to_excel(self.save_folder + 'local_results.xlsx', index=False)

    def _clients_training(self, sampled_clients, r_idx):
        """Conduct local training and get trained local models' weights"""

        updated_local_weights, client_sizes, scores = [], [], []
        round_results = {}

        server_weights = self.model.state_dict()
        server_optimizer = self.optimizer.state_dict()

        # Client training stage
        for client_idx in sampled_clients:
            self._set_client_data(client_idx, r_idx)
            # Download global
            self.client.download_global(server_weights, server_optimizer)

            # Local training
            local_results, local_size, score = self.client.train()
            scores.append(score)
            # save local results
            df = pd.DataFrame.from_dict([local_results])
            df['Round'] = r_idx
            self.local_clients_info = pd.concat([self.local_clients_info, df])
            # Upload locals
            updated_local_weights.append(self.client.upload_local())

            # Update results
            round_results = self._results_updater(round_results, local_results)
            client_sizes.append(local_size)

            # Reset local model
            self.client.reset()

        return updated_local_weights, client_sizes, round_results, scores

    def _get_local_criterion(self):
        criterion = weighted_loss()
        return criterion

    def _set_client_data(self, client_idx, noise=0):
        """Assign local client datasets."""
        self.client.datasize = self.data_distributed["local"][client_idx]["datasize"]
        self.client.trainloader = self.data_distributed["local"][client_idx]["train"]
        self.client.local_testloader = self.data_distributed["local"][client_idx]["test"]
        self.client.testsize = self.data_distributed["local"][client_idx]["test_size"]

        self.client.global_testloader = self.data_distributed["global"]["test"]
        self.client.global_test_size = self.data_distributed["global"]["test_size"]
        self.client.current_client = client_idx
        self.client.class_distribution = self.data_distributed["local"][client_idx]["class_distribution"]