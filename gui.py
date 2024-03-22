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
    V = 25
    K = 3
    filename = "instance.dzn"
    save_instances(num_instances, V, K, filename)
    messagebox.showinfo("Generate Instances", f"Generated {num_instances} instances.")
    instances_generated = True

def solve_instances():
    if not instances_generated:
        messagebox.showinfo("Error", "Please generate instances first.")
        return
    
    # Use the user-selected .mzn file path
    model_path = file_entry.get()
    if not model_path:
        messagebox.showinfo("Error", "Please select a .mzn file.")
        return
    if not model_path.endswith(".mzn"):
        messagebox.showinfo("Error", "The selected file must be a .mzn file.")
        return

    # Similarly, update the instance path if needed
    instance_path = os.path.join('instances', '0instance.dzn')

    model = minizinc.Model(model_path)
    gecode_solver = minizinc.Solver.lookup("gecode")
    instance = minizinc.Instance(gecode_solver, model)
    
    # If the model requires a data file, add it here
    # instance.add_file(instance_path)

    result = instance.solve()
    
    if result.solution is not None:
        messagebox.showinfo("Solve Instances", f"Solution:\n{result}")
    else:
        messagebox.showinfo("Solve Instances", "No solution found.")

def select_file():
    # Update to only accept .mzn files
    file_path = filedialog.askopenfilename(filetypes=[("MiniZinc files", "*.mzn")])
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)


root = tk.Tk()
root.title("MiniZinc Instance Solver")

# Options for dropdown menus
options1 = ["GNN", "Supervised", "Reinforcement"]

# Define tkinter variable for dropdown selection
dropdown_var1 = tk.StringVar(root)
dropdown_var1.set("GNN")

# Create an entry widget to display the file path
file_entry = tk.Entry(root, width=50)
file_entry.pack(padx=10, pady=5)

# Create a button to select a file
select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack(padx=10, pady=5)

# Create buttons for generating and solving instances
generate_button = tk.Button(root, text="Generate Instances", command=generate_instances)
generate_button.pack(padx=10, pady=5)

solve_button = tk.Button(root, text="Solve Instances", command=solve_instances)
solve_button.pack(padx=10, pady=5)

# Create a dropdown menu for model selection
dropdown1 = tk.OptionMenu(root, dropdown_var1, *options1)
dropdown1.pack(padx=10, pady=10)

root.mainloop()
