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
from sklearn.cluster import KMeans

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Suppress all user warnings
warnings.simplefilter(action='ignore', category=UserWarning)


import torch.nn.functional as F

# import load_dataset as ld
import pandas as pd

# Set random seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="batch size", type=int, default=100)
parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
parser.add_argument("--num_epochs", help="number of epochs", type=int, default=20)
parser.add_argument("--num_workers", help="number of workers", type=int, default=20)
parser.add_argument("-f", help="f parameter", type=int, default=0)
parser.add_argument("-m", "--mal_workers", help="number of malicious workers", type=int, default=10)
parser.add_argument("--cuda", action="store_true", help="use CUDA if available")
parser.add_argument("--th", help="trust index", type=float, default=1.0)


args = parser.parse_args()

NUM_CLIENTS = args.num_workers
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
LEARNING_RATE = args.lr
F_PARAM = args.f
MAL_WORKERS = args.mal_workers
DEVICE = "cpu"
T_H = args.th

history_dict = {}
for cid in range(NUM_CLIENTS):
    history_dict[cid] = []

data_dict = {'Epoch': [], 'Accuracy': [], 'Loss': [], 'Duration': []}

mal_client_dict = {'mal' : []}

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
    return [param.grad.clone() for param in model.parameters() if param.requires_grad and param.grad is not None]

def extract_normallized_gradients(model):
    gradients = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad.clone()
            grad_norm = torch.norm(grad)
            if grad_norm > 0:
                grad = grad / grad_norm
            gradients.append(grad)
    return gradients

def krum_gradients(model, worker_models):
    # First, extract gradients from each worker model
    worker_gradients = [extract_normallized_gradients(worker) for worker in worker_models]

    normalized_gradients = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0) for x in worker_gradients]

    proper_gradients = [extract_gradients(worker) for worker in worker_models]

    proper_grad_format = [torch.cat([xx.reshape(-1, 1) for xx in x], dim=0) for x in proper_gradients]
    
    modified_gradients = []
    real_grads = []
    ########## GAUSSIAN ATTACK ##########
    # for i in range(len(normalized_gradients)):
    #     if i < MAL_WORKERS:
    #         noise = torch.normal(0, 5, normalized_gradients[i].size())
    #         modified_gradients.append(normalized_gradients[i] + noise)
    #     else:
    #         modified_gradients.append(normalized_gradients[i])

    ######### COMBINED GAUSSIAN + BIT FLIP ATTACK ##########
    # for i in range(len(normalized_gradients)):
    #     if i < MAL_WORKERS:
    #         noise = torch.normal(0, 1, normalized_gradients[i].size())
    #         modified_gradients.append(-normalized_gradients[i] + noise)
    #     else:
    #         modified_gradients.append(normalized_gradients[i])

    ########## BIT FLIP ATTACK ##########
    for i in range(len(normalized_gradients)):
        if i < MAL_WORKERS:
            modified_gradients.append(normalized_gradients[i] * -1)
            real_grads.append(proper_grad_format[i]*-1)
        else:
            modified_gradients.append(normalized_gradients[i])
            real_grads.append(proper_grad_format[i])

    # for i in range(len(normalized_gradients)):
    #     if i < MAL_WORKERS:
    #         if torch.rand(1).item() < 0.5:
    #             modified_gradients.append(normalized_gradients[i] * -1)
    #             real_grads.append(proper_grad_format[i]*-1)
    #         else:
    #             modified_gradients.append(normalized_gradients[i])
    #             real_grads.append(proper_grad_format[i])
    #     else:
    #         modified_gradients.append(normalized_gradients[i])
    #         real_grads.append(proper_grad_format[i])

    v = torch.cat(modified_gradients, dim=1)

    # Calculate Krum Scores
    scores_res = []
    for grad in modified_gradients:
        workers = v.shape[1] - F_PARAM - 2 #0
        sorted_distance = torch.sum((v - grad) ** 2, dim=0).sort().values
        scores_res.append(sorted_distance[1:(1 + workers)].sum().item())
    scores_res = torch.tensor(scores_res)

    # print("KRUM SCORE: ",scores_res)
    
    first_elements = scores_res

    # Find the minimum and maximum values among the first elements
    min_val = min(first_elements)
    max_val = max(first_elements)

    trust_scores = []

    for i in range(0, len(scores_res)):
        score = scores_res[i]
        cid = i
        
        # Check max_val and min_val
        if max_val == min_val:
            normalized_score = 0
        else:
            normalized_score = (score - min_val) / (max_val - min_val)

        history_dict[cid].append(normalized_score)
        trust_scores.append((normalized_score, cid))
        
        hist_list = history_dict[cid]
        
        if len(hist_list) >= 50:
            roll_avg = torch.mean(torch.tensor(hist_list[len(hist_list)-100:len(hist_list)]))
            trust_scores[i] = (trust_scores[i][0] + (T_H * roll_avg), cid)


    temp_scores = scores_res


    kmeans = KMeans(n_clusters=2, random_state=0).fit(temp_scores.reshape(-1, 1)) #view(-1, 1)
    

    indices_zero = np.where(kmeans.labels_ == 0)[0]
    indices_one = np.where(kmeans.labels_ == 1)[0]

    trust_ones = [trust_scores[i] for i in indices_one]
    trust_zeros = [trust_scores[i] for i in indices_zero]

    # print("TRUST ONES: ", trust_ones)
    # print("TRUST ZEROS: ", trust_zeros)
    # print("")
    # print("KMEANS LABELS: ", kmeans.labels_)
    # trust_scores = None
    if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1]:
        trust_scores = trust_zeros
    else:
        trust_scores = trust_ones

    # print(kmeans.cluster_centers_)
    



    mal_client_dict['mal'].append(len(trust_scores))
    _, selected_grad_idx = min(trust_scores)
    selected_grad = real_grads[selected_grad_idx]

    krum_nd = real_grads[selected_grad_idx].view(-1)
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
        # for batch in testloader:
        for (data, target) in tqdm(loader, type):


            data, target = data.to(device=DEVICE), target.to(device=DEVICE)
            output = global_model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader)
    test_accuracy = correct / len(loader)
    print(f"Epoch {epoch+1}. {type} accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}, Time: {time.time()-tic:.2f}s")
    data_dict['Epoch'].append(epoch+1)
    data_dict['Accuracy'].append(test_accuracy)
    data_dict['Loss'].append(test_loss)
    data_dict['Duration'].append(time.time()-tic)


for epoch in range(EPOCHS):
    tic = time.time()
    
    train()
    evaluate(loader=valloader, type="Validation")
evaluate(loader=testloader, type="Testing")

df = pd.DataFrame(data_dict)
mal_df = pd.DataFrame(mal_client_dict)

# Write DataFrame to CSV
# df.to_csv(f'./knn_data/output_strat-trust-explol-krumknn_ep{EPOCHS}_f{F_PARAM}_att-GAUSSIAN_n{NUM_CLIENTS}_m{MAL_WORKERS}.csv', index=False)
mal_df.to_csv(f'./knn_data/mal_clients_strat-trust-expvary-krumknn_ep{EPOCHS}_f{F_PARAM}_att-GAUSSIAN_n{NUM_CLIENTS}_m{MAL_WORKERS}.csv', index=False)
# df.to_csv(f'./reruns/output_strat-trust-krum_ep{EPOCHS}_f{F_PARAM}_att-FLIP_n{NUM_CLIENTS}_m{MAL_WORKERS}.csv', index=False)
