import argparse
import torch

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split

import time
import random
import numpy as np
from tqdm import tqdm
import random

import torch.nn.functional as F

import pandas as pd

# Set random seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="batch size", type=int, default=100)
parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
parser.add_argument("--num_epochs", help="number of epochs", type=int, default=30)
parser.add_argument("--num_workers", help="number of workers", type=int, default=20)
parser.add_argument("-f", help="f parameter", type=int, default=8)
parser.add_argument("-m", "--mal_workers", help="number of malicious workers", type=int, default=8)
parser.add_argument("--cuda", action="store_true", help="use CUDA if available")

args = parser.parse_args()

NUM_CLIENTS = args.num_workers
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.lr
F_PARAM = args.f
MAL_WORKERS = args.mal_workers
DEVICE = "cpu"

data_dict = {'Epoch': [], 'Accuracy': [], 'Loss': [], 'Duration': []}

frequency = {}
bad_freq = {}

for i in range(0, MAL_WORKERS):
    bad_freq[i] = 0

for i in range(0, NUM_CLIENTS):
    frequency[i] = 0

class Net(nn.Module):
    '''
    Created a Client Model which goes through 3 layers
    - First layer is a linear layer based on the dimension of the mnist images (28x28)
    - Second layer takes the output of the first in order to break down dimensions
    - Final layer looks to output a number bw 0-9 i.e. 10 outputs
    
    '''
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
    

global_model = Net().to(device=DEVICE)


criterion = nn.NLLLoss() 

train_dataset = datasets.MNIST('./data', train=True, download=True,
                            transform = transforms.Compose(
                            [
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))  
                            ]))

test_dataset = datasets.MNIST('./data', train=False,
                              transform = transforms.Compose(
                            [
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))  
                            ]))
print(type(train_dataset))
num_train = len(train_dataset)
num_val = int(0.2 * num_train)
num_train -=num_val
train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



def extract_gradients(model):
    return [param.grad.view(-1) for param in model.parameters() if param.requires_grad and param.grad is not None]

def krum_gradients(model, worker_models):
    # First, extract gradients from each worker model
    worker_gradients = [extract_gradients(worker) for worker in worker_models]

    normalized_gradients = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0) for x in worker_gradients]

    modified_gradients = []
    ########## BIT FLIP ATTACK ##########
    # Commented out if-statement code has logic to alternate between 
    # constant and variable maliciousness.
    mods = []
    for i in range(len(normalized_gradients)):
        if i < MAL_WORKERS:
            # if torch.rand(1).item() < 0.5:
            mods.append(i)
            modified_gradients.append(normalized_gradients[i] * -1)
            # else:
            #     modified_gradients.append(normalized_gradients[i])
        else:
            modified_gradients.append(normalized_gradients[i])

    v = torch.cat(modified_gradients, dim=1)

    # Calculate Krum Scores
    scores_res = []
    for grad in modified_gradients:
        workers = v.shape[1] - F_PARAM - 2
        sorted_distance = torch.sum((v - grad) ** 2, dim=0).sort().values
        scores_res.append(sorted_distance[1:(1 + workers)].sum().item())
    scores_res = torch.tensor(scores_res)

    selected_grad_idx = scores_res.argmin(dim=0).item()

    if selected_grad_idx in mods:
        bad_freq[selected_grad_idx] +=1

    frequency[selected_grad_idx] +=1

    krum_nd = modified_gradients[selected_grad_idx].view(-1)
    idx = 0
    for param in model.parameters():
        if param.requires_grad:
            num_param_elements = param.numel()
            param_update_slice = krum_nd[idx:idx + num_param_elements]
            param_update = param_update_slice.view(param.size())
            with torch.no_grad():
                param.data -= LEARNING_RATE * param_update
            idx += num_param_elements



def train():
    global_model.train()
    worker_models = []
    for batch_idx, (data,target) in enumerate(tqdm(trainloader, desc="Training")):
        worker_model = Net().to(device=DEVICE)
        worker_model.load_state_dict(global_model.state_dict())
        
        output = worker_model(data)
        loss = criterion(output, target)
        loss.backward()
        worker_models.append(worker_model)
        if (len(worker_models) == NUM_CLIENTS):
            krum_gradients(global_model, worker_models)
            worker_models = []

def evaluate(loader, type):
    global_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in tqdm(loader, type):
            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            output = global_model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader)
    test_accuracy = correct / len(loader)
    final_time = time.time()-tic
    print(f"Epoch {epoch+1}. {type} accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}, Time: {final_time:.2f}s")
    data_dict['Epoch'].append(epoch+1)
    data_dict['Accuracy'].append(test_accuracy)
    data_dict['Loss'].append(test_loss)
    data_dict['Duration'].append(round(final_time, 4))



for epoch in range(EPOCHS):
    tic = time.time()
    
    train()
    evaluate(loader=valloader, type="Validation")
evaluate(loader=testloader, type="Testing")

print(f'FREQUENCIES: {frequency}')
print(f'BAD FREQUENCIES: {bad_freq}')
df = pd.DataFrame(data_dict)

# Write DataFrame to CSV
df.to_csv(f'/home/lileshk2/krum_data/output_strat-krum_ep{EPOCHS}_f{F_PARAM}_att-GAUSSIAN_n{NUM_CLIENTS}_m{MAL_WORKERS}.csv', index=False)

