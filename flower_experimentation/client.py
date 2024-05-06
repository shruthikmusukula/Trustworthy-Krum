import argparse
import warnings
from collections import OrderedDict

import flwr as fl
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import torch.optim as optim
import torchvision.transforms as transforms


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
NUM_CLIENTS = 3
DATASET_NAME = "mnist"
BATCH_SIZE = 32
NUM_EPOCHS=1


class ClientModel(nn.Module):
    '''
    Created a Client Model which goes through 3 layers
    - First layer is a linear layer based on the dimension of the mnist images (28x28)
    - Second layer takes the output of the first in order to break down dimensions
    - Final layer looks to output a number bw 0-9 i.e. 10 outputs
    
    '''
    def __init__(self):
        super(ClientModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  
        self.fc2 = nn.Linear(128, 64)   
        self.fc3 = nn.Linear(64, 10)      

    def forward(self, x):
        x = x.view(-1, 28*28)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)  
        return F.log_softmax(x, dim=1)  # Return log-probability
    

def train(model, train_loader, epochs, criterion, optimizer, train_idx):
    '''
     Basic train function which takes a single partitioned dataset, 
     and is able to look at batched output, loss, and step in the right direction.
     Might want to implement tqdm library
    '''

    model.train()  

    batch = None
    for i, b in enumerate(train_loader):
        if i == train_idx:
            batch = b
    data = batch["image"]
    target = batch["label"]

    optimizer.zero_grad() 
    output = model(data) 
    loss = criterion(output, target)  
    loss.backward()



def validate(model, val_loader, criterion):
    '''
    Validation step in our pipeline, for model to reinforce against new dataset (stray away from overfitting)
    '''
    model.eval()

    val_loss = 0
    correct = 0
    with torch.no_grad():  
        for batch in tqdm(val_loader, "Validation"):
            data = batch["image"]
            target = batch["label"]

            output = model(data)
            val_loss += criterion(output, target).item()  
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)\n')
    accuracy = correct / len(val_loader.dataset)
    return val_loss, accuracy


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

# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "-p",
    "--partition-id",
    choices=[0, 1, 2],
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)

# Get number of clients
parser.add_argument(
    "-n",
    "--nclients",
    choices=range(1,21),
    metavar="[1-20]",
    required=True,
    type=int,
    help="Number of client processes for the current simulation.",
)

# Get main server IP address
parser.add_argument(
    "-s",
    "--server-ip",
    required=True,
    type=str,
    help="IP Address of the main FL server for the current simulation.",
)

# Parse all CLI arguments
cid = parser.parse_args().partition_id
NUM_CLIENTS = parser.parse_args().nclients
server_address = parser.parse_args().server_ip

# Load model and data (simple CNN, CIFAR-10)
model= ClientModel()
trainloader, valloader, _ = load_dataset(cid=cid)
criterion = nn.NLLLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.train_idx = 0
        self.epoch = 0

    def get_parameters(self):
        grad_collect = []
        for param in model.parameters():
            if param.requires_grad:
                # Copy the gradient if it exists
                if param.grad is not None:
                    grad = param.grad.clone()

                    grad_collect.append(grad)
                
        return grad_collect  

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(model, trainloader, epochs=NUM_EPOCHS, criterion=criterion, optimizer=optimizer, train_idx=self.train_idx)
        self.train_idx +=1
        if (self.train_idx == len(trainloader)):
            self.train_idx = 0
            self.epoch +=1

        return self.get_parameters(), len(trainloader.dataset), {'done' : self.epoch}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = 0
        accuracy = 0
        if self.train_idx == 0:
            loss, accuracy = validate(model, valloader, criterion=criterion)
        return float(loss), len(valloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address=server_address,
    client=FlowerClient().to_client(),
)

