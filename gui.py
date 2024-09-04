import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import random
import minizinc 
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import joblib
import subprocess


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
    try:
        num_instances = int(num_instances_entry.get())
        if num_instances <= 0:
            raise ValueError("Number of instances must be a positive integer.")
    except ValueError as e:
        messagebox.showerror("Invalid input", "Please enter a valid number of instances.")
        return

    # Note: when V= 25, solver takes under 30 minute to solve, when V=20 solver takes under 1 minute to solve
    V = 15
    K = 3
    filename = "instance.dzn"
    save_instances(num_instances, V, K, filename)
    messagebox.showinfo("Generate Instances", f"Generated {num_instances} instances.")
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
    except ValueError as e:
        messagebox.showerror("Invalid input", "Please enter a valid number of instances to solve.")
        return

    dzn_files = sorted([f for f in os.listdir("instances") if f.endswith(".dzn")])[:num_solve_instances]
    
    if num_solve_instances > len(dzn_files):
        messagebox.showerror("Invalid input", f"Number of instances to solve exceeds the available generated instances ({len(dzn_files)}).")
        return

    try:
        train_percentage = int(train_percentage_entry.get())
        if not 0 < train_percentage < 100:
            raise ValueError("Percentage must be between 1 and 99.")
    except ValueError as e:
        messagebox.showerror("Invalid input", "Please enter a valid percentage between 1 and 99.")
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

def train_model():
    from trainer import process_data 

    # Process the data and check if it was successful
    if process_data():
        messagebox.showinfo("Success", "Data processed successfully and is ready for training.")
    else:
        messagebox.showerror("Error", "Failed to process data.")

def select_file():
    # Update to only accept .mzn files
    file_path = filedialog.askopenfilename(filetypes=[("MiniZinc files", "*.mzn")])
    if file_path:
        file_entry_mzn.delete(0, tk.END)
        file_entry_mzn.insert(0, file_path)

def analyze_data(root):
    # File selection
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        messagebox.showinfo("Error", "Please select a dataset.")
        return

    # Data handling
    data = pd.read_csv(file_path)
    data['subset_exists'] = data['subset_exists'].astype('category')
    
    try:
        train_percentage = int(train_percentage_entry.get())
        if not 0 < train_percentage < 100:
            raise ValueError("Percentage must be between 1 and 99.")
    except ValueError as e:
        messagebox.showerror("Invalid input", "Please enter a valid percentage between 1 and 99.")
        return

    test_percentage = 100 - train_percentage

    train_data, test_data = train_test_split(data, test_size=test_percentage/100, random_state=1148, stratify=data['subset_exists'])

    # User selects a model from the drop down options
    selected_model = dropdown_var1.get()

    # Models
    if selected_model == "Decision tree model":
         # Read hyperparameters from the input fields
        criterion = criterion_var.get()
        max_depth = None if max_depth_entry.get() == '' else int(max_depth_entry.get())
        min_samples_split = int(min_samples_split_entry.get())
        min_samples_leaf = int(min_samples_leaf_entry.get())

        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        model_name = "Decision Tree"
    elif selected_model == "Xgboost tree model":
        model = XGBClassifier()
        model_name = "XGBoost"
    elif selected_model == "Random forest model":
        model = RandomForestClassifier(n_estimators=1000, max_features=2)
        model_name = "Random Forest"

    # Training
    model.fit(train_data.drop('subset_exists', axis=1), train_data['subset_exists'])
    predictions = model.predict(test_data.drop('subset_exists', axis=1))

    # Evaluate and display model in a new window
    evaluate_model(model, test_data.drop('subset_exists', axis=1), test_data['subset_exists'], model_name)

def export_model(model, model_name):
    save_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl"), ("Joblib files", "*.joblib")])
    if not save_path:
        messagebox.showerror("Error", "No file selected for saving the model.")
        return

    # Save the model
    if save_path.endswith(".pkl"):
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
    elif save_path.endswith(".joblib"):
        joblib.dump(model, save_path)

    messagebox.showinfo("Success", f"Model saved successfully to {save_path}.")

def evaluate_model(model, test_features, test_labels, model_name):
    probabilities = model.predict_proba(test_features)[:, 1]
    fpr, tpr, _ = roc_curve(test_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(test_labels, probabilities)
    average_precision = average_precision_score(test_labels, probabilities)

    # Create a new window
    plot_window = tk.Toplevel()
    plot_window.title("Analysis Results")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot ROC Curve
    ax1.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax1.plot([0, 1], [0, 1], linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve - {model_name}')
    ax1.legend(loc="lower right")

    # Plot Precision-Recall Curve
    ax2.plot(recall, precision, label=f'AP = {average_precision:.2f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve - {model_name}')
    ax2.legend(loc="lower right")

    # Display the plot in the new window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Save button
    save_button = tk.Button(plot_window, text="Save Plots", command=lambda: save_plots(fig, model_name))
    save_button.pack(side=tk.BOTTOM)
    
    # Export model button
    export_button = tk.Button(plot_window, text="Export Model", command=lambda: export_model(model, model_name))
    export_button.pack(side=tk.RIGHT, padx=10, pady=10)

    messagebox.showinfo("Analysis Complete", "Analysis displayed in a new window.")

def save_plots(fig, model_name):
    output_dir = 'analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Save figure
    fig.savefig(os.path.join(output_dir, f'{model_name}_analysis_plots.png'))
    messagebox.showinfo("Save Successful", f"Plots saved in '{output_dir}' folder.")

def run_neural_script():
    try:
        subprocess.run(["python", "neural.py"], check=True)
        messagebox.showinfo("Success", "neural.py ran successfully.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to run neural.py: {e}")

root = tk.Tk()
root.title("MiniZinc Instance Solver")

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


solve_button = tk.Button(root, text="Solve Instances", command=solve_instances)
solve_button.pack(padx=10, pady=5)

solve_button = tk.Button(root, text="Train Model", command=train_model)
solve_button.pack(padx=10, pady=5)

# Button to run neural.py script
neural_button = tk.Button(root, text="Run Neural Script", command=run_neural_script)
neural_button.pack(padx=10, pady=5)


# Initialize the progress bar
progress = ttk.Progressbar(root, orient="horizontal", length=200, mode='determinate')
progress.pack(padx=10, pady=20)

# Drop down values
dropdown = tk.OptionMenu(root, dropdown_var1, *options1)
dropdown.pack()


frame = tk.Frame(root)
frame.pack(pady=20, padx=20)

# Entry for percentage of instances to train
train_percentage_label = tk.Label(root, text="Percentage of Instances to Train:")
train_percentage_label.pack(padx=10, pady=5)
train_percentage_entry = tk.Entry(root, width=10)
train_percentage_entry.pack(padx=10, pady=5)

# Criterion for Decision Tree
tk.Label(frame, text="Criterion:").grid(row=4, column=0, sticky=tk.W)
criterion_var = tk.StringVar(value='gini')
tk.OptionMenu(frame, criterion_var, 'gini', 'entropy').grid(row=4, column=1, sticky=tk.W)

# Max Depth for Decision Tree
tk.Label(frame, text="Max Depth:").grid(row=5, column=0, sticky=tk.W)
max_depth_entry = tk.Entry(frame)
max_depth_entry.grid(row=5, column=1)

# Min Samples Split for Decision Tree
tk.Label(frame, text="Min Samples Split:").grid(row=6, column=0, sticky=tk.W)
min_samples_split_entry = tk.Entry(frame)
min_samples_split_entry.grid(row=6, column=1)

# Min Samples Leaf for Decision Tree
tk.Label(frame, text="Min Samples Leaf:").grid(row=7, column=0, sticky=tk.W)
min_samples_leaf_entry = tk.Entry(frame)
min_samples_leaf_entry.grid(row=7, column=1)

# Model selection
tk.Label(frame, text="Select Model:").grid(row=8, column=0, sticky=tk.W)
dropdown_var1 = tk.StringVar(value='Decision tree model')
model_menu = tk.OptionMenu(frame, dropdown_var1, 'Decision tree model', 'Xgboost tree model', 'Random forest model')
model_menu.grid(row=8, column=1, sticky=tk.W)

# Analyze button
analyze_button = tk.Button(root, text="Analyze Data", command=lambda: analyze_data(root))
analyze_button.pack(padx=10, pady=5)
root.mainloop()
