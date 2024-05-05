import os
import torch
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.optim as optim

#optional
import numpy as np
import pandas as pd



class KidneyExchangeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(KidneyExchangeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        # This method should return a list of all files which are in the 'raw' directory.
        return ['0instance.dzn', '0instance_solution.txt', 'ccmcp.mzn']

    @property
    def processed_file_names(self):
        # This method should return a list of all files which have been processed.
        return ['data.pt']

    def download(self):
        # This method downloads the dataset to the 'raw' directory.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        
        # Example: Load and parse each graph instance and its solution
        # You'll need to implement the parsing logic here

        for data in data_list:
            torch.save(data, os.path.join(self.processed_dir, 'data.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        # Load processed data from disk.
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
