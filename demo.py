import minizinc
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# define the MiniZinc solver
solver = minizinc.Solver.lookup("gurobi")

def split_data(solved_instances, train_ratio=0.8, seed=None):
    if seed is not None:
        random.seed(seed)

    random.shuffle(solved_instances)

    num_instances = len(solved_instances)
    num_train = int(num_instances * train_ratio)

    training_instances = solved_instances[:num_train]
    testing_instances = solved_instances[num_train:]

def prepare_data(solved_instance):
    # assuming solved_instance is a dictionary with 'input' and 'output' keys
    input_data = solved_instance['input']
    target_data = solved_instance['output']

    # preprocess input data
    inputs = input_data
    # preprocess target data
    targets = target_data

    # convert to PyTorch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)

    return inputs, targets

# load MiniZinc instances
instances_dir = '4701/instances'
mz_model = minizinc.model.Model('4701/ccmcp.mzn')
instance_paths = os.listdir(instances_dir)
minizinc_instances = []
for instance_path in instance_paths:
    instance = minizinc.instance.Instance(solver, mz_model)
    instance.add_file(instances_dir+'/'+instance_path, parse_data=True)
    minizinc_instances.append(instance)


# solve MiniZinc instances
solved_instances = []
filename = "results.txt"
for i, instance in enumerate(minizinc_instances):
    print('solving', i)
    result = instance.solve()
    with open('4701/results/'+str(i)+filename, "w") as f:
        f.write(str(result))
        f.write("\n")
    solved_instances.append(result)

breakpoint()

# prepare training and testing data
training_instances, testing_instances = split_data(solved_instances)

# define PyTorch model
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# train the model
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
num_epochs = 50000

for epoch in range(num_epochs):
    for instance in training_instances:
        # prepare input and target from the instance
        inputs, targets = prepare_data(instance)

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# test the model
with torch.no_grad():
    for instance in testing_instances:
        inputs, targets = prepare_data(instance)
        outputs = model(inputs)
        breakpoint()
        # evaluate the model's performance

