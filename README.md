# FedKPer

## Abstract
Federated learning (FL) holds great potential for medical applications. However, statistical heterogeneity across healthcare institutions poses a major challenge for FL, as the global model struggles both to generalize across unseen patient populations and to adapt to the unique data distributions of individual hospitals. This heterogeneity also exacerbates forgetting at both the global and local level, resulting in previous learned patient patterns to be misclassified after model updates. While prior work has largely treated generalization and personalization as separate challenges, we show that a better balance between the two can be achieved through selective alignment with the global model and a modified aggregation scheme, which together mitigate the effects of statistical heterogeneity. Specifically, we introduce FedKPer, which introduces knowledge personalization into the training stage of each local device. Afterwards, generalization is considered via the global model aggregation process, where local updates that are reliable and label-diverse are emphasized. We evaluate the performance of FedKPer, devising additional metrics that relate to common consequences of forgetting. Overall, we demonstrate FedKPer improves the generalization-personalization trade-off without sacrificing retention.

## Run Instructions
Federated learning is an iterative process. It takes place across multiple communication rounds. At each round, we sample a certain percentage of all clients. We are easily able to set these parameters using this codebase. For instance, say we want to run the FedAvg algorithm. We can set up a bash script to contain this line:

```
python3 [path-to-repo-location]/main.py --seed=1 --partition_method="dirichlet" --partition_alpha=0.1 --n_rounds=200 --batch_size=12 --n_clients=20 --root='path-to-dataset' --dataset_name='BloodMNIST' --model_name='fedavg_cifar' --base_folder='path-to-results-folder' --root_path='path-to-your-FL-repo' --sample_ratio=0.1 --date='enter-date-here' --config_path="/config/fedkper.json"
```

In the above example, --sample_ratio is the percentage of clients sampled each round. --n_rounds is the number of total communication rounds. --n_clients is the total number of clients created. You can control the exact algorithm you are running by changing the --config_path.

In federated learning, we also simulate label heterogeneity experiments, where we purposefully make the clients have different label distributions. For instance, maybe client 0 has classes 0 and 1, while client 1 has classes 2 and 3. Clients having heterogeneous label distributions tends to cause the performance of FL algorithms to deteriorate. One of the ways we simulate this data heterogeneity is via a Dirichlet distribution (--partition_method='dirichlet'), which is controlled by an alpha parameter (--partition_alpha) that makes the client partition more heterogeneous.
