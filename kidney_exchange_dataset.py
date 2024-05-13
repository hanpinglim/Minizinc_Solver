import os
import torch
from torch_geometric.data import Dataset, Data
import re

class KidneyExchangeDataset(Dataset):
    def __init__(self, root, processed_dir=None, transform=None, pre_transform=None):
        super(KidneyExchangeDataset, self).__init__(root, transform=transform, pre_transform=pre_transform)
        # Set up the directory for processed data
        self.processed_dir = processed_dir if processed_dir else os.path.join(root, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)

    @property
    def processed_file_names(self):
        # List all .txt files in the processed directory that have solutions
        return [f for f in os.listdir(self.processed_dir) if f.endswith('_solution.txt')]

    def process(self):
        # Read data into a list of `Data` objects.
        for filename in self.processed_file_names:
            file_path = os.path.join(self.processed_dir, filename)
            with open(file_path, 'r') as file:
                content = file.read()
            try:
                data = self.extract_and_create_data(content)
                torch.save(data, os.path.join(self.processed_dir, filename.replace('_solution.txt', '.pt')))
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    def extract_and_create_data(self, content):
        K = int(re.search(r"K = (\d+);", content).group(1))
        V = int(re.search(r"V = (\d+);", content).group(1))
        edge_weights = eval(re.search(r"edge_weight = (\[\|[\s\S]+\|]);", content).group(1))
        succ = eval(re.search(r"succ = array1d\(1\.\.(\d+), (\[[\s\S]+\])\);", content).group(2))
        cycle = eval(re.search(r"cycle = array1d\(1\.\.(\d+), (\[[\s\S]+\])\);", content).group(2))
        objective = int(re.search(r"objective = (\d+);", content).group(1))
        return self.create_graph_data(V, edge_weights, succ, cycle, objective)

    def create_graph_data(self, V, edge_weights, succ, cycle, objective):
        edge_index = []
        edge_attr = []
        for i in range(V):
            for j in range(V):
                if edge_weights[i][j] != 0:
                    edge_index.append([i, j])
                    edge_attr.append(edge_weights[i][j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        node_labels = torch.tensor(succ, dtype=torch.long)
        cycle_tensor = torch.tensor(cycle, dtype=torch.long)
        data = Data(edge_index=edge_index, edge_attr=edge_attr, y=node_labels, objective=objective, cycle=cycle_tensor)
        return data

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
