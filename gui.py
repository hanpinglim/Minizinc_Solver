import tkinter as tk
from tkinter import filedialog, messagebox
import random
import os  


def generate_instance(V, K):
    edge_weight = '[|'
    for i in range(V):
        for j in range(V):
            if i == j:
                edge_weight += '0,'
            else:
                weight = random.randint(-100, 100)
                edge_weight += str(weight) + ','
        edge_weight += '|'
    edge_weight += ']'

    instance = f"""
K = {K};
V = {V};
edge_weight = {edge_weight};
"""
    return instance

def save_instances(num_instances, V, K, filename):
    # Check if the directory exists, if not create it
    if not os.path.exists('instances'):
        os.makedirs('instances')

    for i in range(num_instances):
        with open(os.path.join('instances', str(i) + filename), "w") as f:
            instance = generate_instance(V, K)
            f.write(instance)
            f.write("\n")

def generate_instances():
    num_instances = 1000
    V = 25
    K = 3
    filename = "instance.dzn"
    save_instances(num_instances, V, K, filename)
    messagebox.showinfo("Generate Instances", f"Generated {num_instances} instances and saved to {filename}")

def solve_instances():
    # Your code to solve instances
    messagebox.showinfo("Solve Instances", "Instances solved successfully!")
    
def train_model():
    # Your code to solve instances
    messagebox.showinfo("Train Model", "Training Model")

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

root = tk.Tk()
root.title("MiniZinc Instance Solver")

# Options for dropdown menus
options1 = ["GNN", "Supervised", "reinforcement"]

# Define tkinter variable for dropdown selection
dropdown_var1 = tk.StringVar(root)

# Set default value for the dropdown
dropdown_var1.set("GNN")

# Create an entry widget to display the file path
file_entry = tk.Entry(root, width=50)
file_entry.pack(padx=10, pady=5)

# Create a button to select a file
select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack(padx=10, pady=5)

# Create buttons for generating and solving instances, and training the model
generate_button = tk.Button(root, text="Generate Instances", command=generate_instances)
generate_button.pack(padx=10, pady=5)

solve_button = tk.Button(root, text="Solve Instances", command=solve_instances)
solve_button.pack(padx=10, pady=5)

# Create dropdown menus and pack them
dropdown1 = tk.OptionMenu(root, dropdown_var1, *options1)
dropdown1.pack(padx=10, pady=10)

train_button = tk.Button(root, text="Train Model", command=train_model)
train_button.pack(padx=10, pady=5)

root.mainloop()
