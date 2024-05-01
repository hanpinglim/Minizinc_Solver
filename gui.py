import tkinter as tk
from tkinter import filedialog, messagebox
import os
import random
import minizinc

# Initialize the flag to False to indicate that instances have not been generated yet
instances_generated = False

def generate_instance(V, K):
    # Generates a single instance with random weights
    edge_weight = '[|'
    for i in range(V):
        for j in range(V):
            if i == j:
                edge_weight += '0,'
            else:
                weight = random.randint(-100, 100)
                edge_weight += str(weight) + ','
        edge_weight = edge_weight[:-1] + '|'
    edge_weight += ']'

    instance = f"""
K = {K};
V = {V};
edge_weight = {edge_weight};
"""
    return instance

def save_instances(num_instances, V, K, filename):
    # Ensure the 'instances' directory exists
    if not os.path.exists('instances'):
        os.makedirs('instances')

    # Save the specified number of instances to files
    for i in range(num_instances):
        with open(os.path.join('instances', str(i) + filename), "w") as f:
            instance = generate_instance(V, K)
            f.write(instance)
            f.write("\n")

def generate_instances():
    global instances_generated
    num_instances = 1000
    V = 20
    K = 3
    filename = "instance.dzn"
    save_instances(num_instances, V, K, filename)
    messagebox.showinfo("Generate Instances", f"Generated {num_instances} instances.")
    instances_generated = True


def solve_instance():
    # Use the user-selected .mzn model file path
    instance_path = filedialog.askopenfilename(initialdir="instances", filetypes=[("Data files", "*.dzn")])
    if instance_path:
        file_entry_dzn.delete(0, tk.END)
        file_entry_dzn.insert(0, instance_path)

    model_path = file_entry_mzn.get()
    if not model_path.endswith(".mzn"):
        messagebox.showinfo("Error", "The selected file must be a .mzn model file.")
        return
    
    instance_path = file_entry_dzn.get()
    if not instance_path:
        messagebox.showinfo("Error", "Please select an instance file.")
        return
    if not instance_path.endswith(".dzn"):  # Assuming .dzn is the instance file type
        messagebox.showinfo("Error", "The selected file must be a .dzn file.")
        return

    model = minizinc.Model(model_path)
    ortools_solver = minizinc.Solver.lookup("com.google.ortools.sat")
    instance = minizinc.Instance(ortools_solver, model)
    instance.add_file(instance_path)

    result = instance.solve()
    
    if result.solution is not None:
        messagebox.showinfo("Solve Instances", f"Solution:\n{result}")
    else:
        messagebox.showinfo("Solve Instances", "No solution found.")


def select_file():
    # Update to only accept .mzn files
    file_path = filedialog.askopenfilename(filetypes=[("MiniZinc files", "*.mzn")])
    if file_path:
        file_entry_mzn.delete(0, tk.END)
        file_entry_mzn.insert(0, file_path)


root = tk.Tk()
root.title("MiniZinc Instance Solver")

# Options for dropdown menus
options1 = ["GNN", "Supervised", "Reinforcement"]

# Define tkinter variable for dropdown selection
dropdown_var1 = tk.StringVar(root)
dropdown_var1.set("GNN")

# Create an entry widget to display the file path
label_mzn = tk.Label(root, text="Constraint Model File (.mzn):")
label_mzn.pack(padx=10, pady=5)
file_entry_mzn = tk.Entry(root, width=50)
file_entry_mzn.pack(padx=10, pady=5)

# Create an entry widget to display the file path
label_dzn = tk.Label(root, text="Instance Data File (.dzn):")
label_dzn.pack(padx=10, pady=5)
file_entry_dzn = tk.Entry(root, width=50)
file_entry_dzn.pack(padx=10, pady=5)

# Create a button to select a file
select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack(padx=10, pady=5)


# Create buttons for generating and solving instances
generate_button = tk.Button(root, text="Generate Instances", command=generate_instances)
generate_button.pack(padx=10, pady=5)

solve_button = tk.Button(root, text="Solve Instances", command=solve_instance)
solve_button.pack(padx=10, pady=5)

# Create a dropdown menu for model selection
dropdown1 = tk.OptionMenu(root, dropdown_var1, *options1)
dropdown1.pack(padx=10, pady=10)

root.mainloop()
