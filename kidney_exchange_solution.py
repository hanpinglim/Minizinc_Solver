import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import random
import minizinc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize the flag to False to indicate that instances have not been generated yet
instances_generated = False

# Set the maximum limits for K and V
MAX_K = 3
MAX_V = 15

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
    
    try:
        K = int(k_entry.get())
        V = int(v_entry.get())
        if K > MAX_K or V > MAX_V:
            raise ValueError(f"K must be <= {MAX_K} and V must be <= {MAX_V}.")
    except ValueError as e:
        messagebox.showerror("Invalid Input", f"Invalid K or V values: {e}")
        return
    
    try:
        num_instances = int(num_instances_entry.get())
        if num_instances <= 0:
            raise ValueError("Number of instances must be a positive integer.")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number of instances.")
        return

    filename = "instance.dzn"
    save_instances(num_instances, V, K, filename)
    messagebox.showinfo("Generate Instances", f"Generated {num_instances} instances with K={K} and V={V}.")
    instances_generated = True

def solve_instances():
    processed_dir = 'processed'
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    model_path = file_entry_mzn.get()
    if not model_path.endswith(".mzn"):
        messagebox.showinfo("Error", "No .mzn model file selected or incorrect file type.")
        return

    try:
        num_solve_instances = int(num_solve_instances_entry.get())
        if num_solve_instances <= 0:
            raise ValueError("Number of instances to solve must be a positive integer.")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid number of instances to solve.")
        return

    dzn_files = sorted([f for f in os.listdir("instances") if f.endswith(".dzn")])[:num_solve_instances]
    
    if num_solve_instances > len(dzn_files):
        messagebox.showerror("Invalid Input", f"Number of instances to solve exceeds the available generated instances ({len(dzn_files)}).")
        return

    try:
        train_percentage = int(train_percentage_entry.get())
        if not 0 < train_percentage < 100:
            raise ValueError("Percentage must be between 1 and 99.")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter a valid percentage between 1 and 99.")
        return

    train_dir = os.path.join(processed_dir, 'train')
    test_dir = os.path.join(processed_dir, 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    train_count = int(num_solve_instances * train_percentage / 100)
    test_count = num_solve_instances - train_count

    progress['maximum'] = len(dzn_files)
    progress['value'] = 0
    failed_instances = []
    for i, file_name in enumerate(dzn_files):
        instance_path = os.path.join("instances", file_name)
        with open(instance_path, 'r') as file:
            instance_data = file.read()
        try:
            model = minizinc.Model(model_path)
            solver = minizinc.Solver.lookup("com.google.ortools.sat")
            instance = minizinc.Instance(solver, model)
            instance.add_file(instance_path)
            result = instance.solve()
            solution_filename = file_name.replace('.dzn', '_solution.txt')
            if i < train_count:
                solution_path = os.path.join(train_dir, solution_filename)
            else:
                solution_path = os.path.join(test_dir, solution_filename)
            if result.solution is not None:
                with open(solution_path, 'w') as file:
                    file.write(instance_data + "\n\n" + str(result))
            else:
                raise Exception("No solution found")
        except Exception as e:
            failed_instances.append(file_name)
            print(f"Failed to solve {file_name}: {e}")
        finally:
            progress['value'] += 1  # Increment progress bar value
            root.update()  # Update GUI to reflect progress

    # Show final message with the number of failed instances and their names
    if failed_instances:
        messagebox.showinfo("Completion", f"Failed to solve {len(failed_instances)} instances: {', '.join(failed_instances)}")
    else:
        messagebox.showinfo("Completion", "All instances processed successfully with no failures.")

def select_file():
    # Update to only accept .mzn files
    file_path = filedialog.askopenfilename(filetypes=[("MiniZinc files", "*.mzn")])
    if file_path:
        file_entry_mzn.delete(0, tk.END)
        file_entry_mzn.insert(0, file_path)

def run_kidney_exchange_gui():
    """Function to launch the Kidney Exchange GUI."""
    global root, progress, file_entry_mzn, num_instances_entry, num_solve_instances_entry, train_percentage_entry, dropdown_var1, k_entry, v_entry

    root = tk.Tk()
    root.title("MiniZinc Instance Solver (Max K = 3, Max V = 15)")

    # Options for dropdown menus
    options1 = ["Decision tree model", "Xgboost tree model", "Random forest model"]

    # Define tkinter variable for dropdown selection
    dropdown_var1 = tk.StringVar(root)
    dropdown_var1.set("Decision tree model")

    # Create an entry widget to display the file path
    label_mzn = tk.Label(root, text="Constraint Model File (.mzn):")
    label_mzn.pack(padx=10, pady=5)
    file_entry_mzn = tk.Entry(root, width=50)
    file_entry_mzn.pack(padx=10, pady=5)

    # Create a button to select a file
    select_button = tk.Button(root, text="Select mzn constraint File", command=select_file)
    select_button.pack(padx=10, pady=5)

    # Entry for K value
    k_label = tk.Label(root, text="Enter K (Max 3):")
    k_label.pack(padx=10, pady=5)
    k_entry = tk.Entry(root, width=10)
    k_entry.pack(padx=10, pady=5)
    
    # Entry for V value
    v_label = tk.Label(root, text="Enter V (Max 15):")
    v_label.pack(padx=10, pady=5)
    v_entry = tk.Entry(root, width=10)
    v_entry.pack(padx=10, pady=5)

    # Entry for number of instances
    num_instances_label = tk.Label(root, text="Number of Instances:")
    num_instances_label.pack(padx=10, pady=5)
    num_instances_entry = tk.Entry(root, width=10)
    num_instances_entry.pack(padx=10, pady=5)

    # Create buttons for generating and solving instances
    generate_button = tk.Button(root, text="Generate Instances", command=generate_instances)
    generate_button.pack(padx=10, pady=5)

    # Entry for number of instances to solve
    num_solve_instances_label = tk.Label(root, text="Number of Instances to Solve:")
    num_solve_instances_label.pack(padx=10, pady=5)
    num_solve_instances_entry = tk.Entry(root, width=10)
    num_solve_instances_entry.pack(padx=10, pady=5)

    # Entry for percentage of instances to train
    train_percentage_label = tk.Label(root, text="Percentage of Instances to Train:")
    train_percentage_label.pack(padx=10, pady=5)
    train_percentage_entry = tk.Entry(root, width=10)
    train_percentage_entry.pack(padx=10, pady=5)

    solve_button = tk.Button(root, text="Solve Instances", command=solve_instances)
    solve_button.pack(padx=10, pady=5)

    # Initialize the progress bar
    progress = ttk.Progressbar(root, orient="horizontal", length=200, mode='determinate')
    progress.pack(padx=10, pady=20)

    # Drop down values
    #dropdown = tk.OptionMenu(root, dropdown_var1, *options1)
    #dropdown.pack()

    root.mainloop()

