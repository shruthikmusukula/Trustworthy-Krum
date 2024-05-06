import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


from flwr_datasets import FederatedDataset
from tqdm import tqdm


DEVICE = "cpu"
DATASET_NAME = "mnist"
BATCH_SIZE = 32
NUM_EPOCHS=1
NUM_CLIENTS = 3



def load_dataset(cid):
    '''Loading DATASET_NAME (mnist) for given cid'''
    fds = FederatedDataset(dataset=DATASET_NAME, partitioners={"train": NUM_CLIENTS})
    print(fds)


    def apply_transforms(batch):

        transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize((0.5,), (0.5,))  
            ]
        )
        batch["image"] = [transform(img) for img in batch["image"]]
        return batch

    
    partition = fds.load_partition(cid, "train")
    partition = partition.with_transform(apply_transforms)
    partition = partition.train_test_split(train_size=0.8)
    trainloader = DataLoader(partition["train"], batch_size=BATCH_SIZE, drop_last=True)
    valloader = DataLoader(partition["test"], batch_size=BATCH_SIZE, drop_last=True)
    
    testset = fds.load_full("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,drop_last=True)
    return trainloader, valloader, testloader


trainloader, valloader, testloader = load_dataset(cid=0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  
        self.fc2 = nn.Linear(128, 64)   
        self.fc3 = nn.Linear(64, 10)      

    def forward(self, x):
        x = x.view(-1, 28*28)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return F.log_softmax(x, dim=1)
       

global_model = Net()
criterion = nn.NLLLoss() 
# Custom Flower Strategy
class FedCustom(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

    def __repr__(self) -> str:
        return "FedCustom"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        )

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        half_clients = n_clients // 2
        standard_config = {"lr": 0.001}
        fit_configurations = []
        for idx, client in enumerate(clients):
            if idx < half_clients:
                fit_configurations.append((client, FitIns(parameters, standard_config)))
            else:
                fit_configurations.append(
                    (client, FitIns(parameters, standard_config))
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}

        gradients = [
            parameters_to_ndarrays(result.parameters) for _, result in results
        ]

        done = results[0][1].metrics['done'] == 1


        gradients_torch = [[torch.from_numpy(arr) for arr in sublist] for sublist in gradients]
        param_list = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0) for x in gradients_torch]
        

        mean_nd = torch.mean(torch.cat(param_list, dim=1), dim=-1)


        idx = 0
        for param in global_model.parameters():
            if param.requires_grad:
               
                num_param_elements = param.numel()
                param_update_slice = mean_nd[idx:idx + num_param_elements]
                param_update = param_update_slice.view(param.size())
                with torch.no_grad():  
                    param.data -= 0.001 * param_update
                idx += num_param_elements


        if done:
            self.evaluate_global_model(server_round)

        updated_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        )
        return updated_parameters, {}
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated
    

    def evaluate(self,
        server_round: int,
        parameters: fl.common.NDArrays,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        loss, accuracy = self.test(global_model, testloader)
        print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
        return loss, {"accuracy": accuracy}
    
    def test(self, model, val_loader):
        '''
        Validation step in our pipeline, for model to reinforce against new dataset (stray away from overfitting)
        '''
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():  
            for batch in tqdm(val_loader, "Testing"):
                data = batch["image"]
                target = batch["label"]

                output = model(data)
                val_loss += criterion(output, target).item()  
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(target.view_as(pred)).sum().item()

        print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n')
        accuracy = correct / len(val_loader.dataset)
        return val_loss, accuracy



    def evaluate_global_model(self, server_round):
        """Evaluate the global model."""
        parameters = self.get_global_model_parameters()
        loss, metrics = self.evaluate(server_round=server_round, parameters=parameters)
        print(f"Global Model - Average Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")

    def get_global_model_parameters(self):
        """Return the global model parameters."""
        net = global_model
        return fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in net.state_dict().items()]
        )

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

def main():
    # Define the Flower strategy
    strategy = FedCustom(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    # Start the Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=len(trainloader)),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

    