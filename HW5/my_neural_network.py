import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, training_X_:np.ndarray, training_y_:np.ndarray, device):
        self.X_ = torch.from_numpy(training_X_).float().to(device)
        self.y_ = torch.from_numpy(training_y_).float().to(device)
        self.len = self.X_.shape[0]
    def __getitem__(self,index):      
        return self.X_[index], self.y_[index]
    def __len__(self):
        return self.len

class MyNeuralNetwork(nn.Module):
    # def __init__(self,input,H,output):
    #     super(NN,self).__init__()
    #     self.linear1=nn.Linear(input,H)
    #     self.linear2=nn.Linear(H,output)

    # def forward(self,x):
    #     x=torch.relu(self.linear1(x))  
    #     x=torch.sigmoid(self.linear2(x))
    #     return x

    def __init__(self, nn_stack):
        super(MyNeuralNetwork, self).__init__()
        self.stack = nn_stack

    def forward(self, x):
        logits = self.stack(x)
        return logits