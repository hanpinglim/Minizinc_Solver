import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import random
import minizinc 
import time
import threading
from minizinc import Model, Instance, Solver

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
    #Note: when V= 25, solver takes under 30 minute to solve, when V=20 solver takes under 1 minute to solve
    V = 15
    K = 3
    filename = "instance.dzn"
    save_instances(num_instances, V, K, filename)
    messagebox.showinfo("Generate Instances", f"Generated {num_instances} instances.")
    instances_generated = True


def solve_instances():
    if not os.path.exists('solved_instances'):
        os.makedirs('solved_instances')

    model_path = file_entry_mzn.get()
    if not model_path.endswith(".mzn"):
        messagebox.showinfo("Error", "No .mzn model file selected or incorrect file type.")
        return

    # Fetching first 100 .dzn files
    dzn_files = sorted([f for f in os.listdir("instances") if f.endswith(".dzn")])[:100]
    total_files = len(dzn_files)

    # Configure progress bar maximum value
    progress['maximum'] = total_files
    progress['value'] = 0  # Reset progress bar to 0 at the start

    failed_instances = []  # List to keep track of failed instances

    for file_name in dzn_files:
        instance_path = os.path.join("instances", file_name)
        
        try:
            model = minizinc.Model(model_path)
            ortools_solver = minizinc.Solver.lookup("com.google.ortools.sat")
            instance = minizinc.Instance(ortools_solver, model)
            instance.add_file(instance_path)

            result = instance.solve()
            
            # Save the solution to a file in the 'solved_instances' directory
            solution_filename = file_name.replace('.dzn', '_solution.txt')
            solution_path = os.path.join('solved_instances', solution_filename)

            if result.solution is not None:
                with open(solution_path, 'w') as file:
                    file.write(str(result))
            else:
                raise Exception("No solution found")
            
        except Exception as e:
            failed_instances.append(file_name)  # Add the file name to the list of failed instances
            print(f"Failed to solve {file_name}: {e}")  # Optionally print or log the error

        finally:
            progress['value'] += 1  # Increment progress bar value
            root.update_idletasks()  # Update GUI to reflect progress

        progress['value'] += 1  # Increment progress bar value
        root.update_idletasks()  # Update GUI to reflect progress

    # Show final message with the number of failed instances and their names
    if failed_instances:
        failed_message = f"Failed to solve {len(failed_instances)} instances: " + ", ".join(failed_instances)
        messagebox.showinfo("Completion", f"All instances processed.\n{failed_message}")
    else:
        messagebox.showinfo("Completion", "All instances processed successfully with no failures.")

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

# Create a button to select a file
select_button = tk.Button(root, text="Select mzn constraint File", command=select_file)
select_button.pack(padx=10, pady=5)

# Create buttons for generating and solving instances
generate_button = tk.Button(root, text="Generate Instances", command=generate_instances)
generate_button.pack(padx=10, pady=5)

solve_button = tk.Button(root, text="Solve Instances", command=solve_instances)
solve_button.pack(padx=10, pady=5)

# Create a dropdown menu for model selection
dropdown1 = tk.OptionMenu(root, dropdown_var1, *options1)
dropdown1.pack(padx=10, pady=10)

# Initialize the progress bar
progress = ttk.Progressbar(root, orient="horizontal", length=200, mode='determinate')
progress.pack(padx=10, pady=20)


root.mainloop()
