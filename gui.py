import tkinter as tk
from tkinter import filedialog, messagebox
import os
import random
import minizinc
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import random
import minizinc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

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
    #Note: when V= 25, solver takes under 20 minute to solve, when V=20 solver takes under 1 minute to solve
    V = 15 
    K = 3
    filename = "instance.dzn"
    save_instances(num_instances, V, K, filename)
    messagebox.showinfo("Generate Instances", f"Generated {num_instances} instances.")
    instances_generated = True


def solve_instance():
    if not os.path.exists('solved_instances'):
        os.makedirs('solved_instances')

    # Use the user-selected .dzn model file path
    instance_path = filedialog.askopenfilename(initialdir="instances", filetypes=[("Data files", "*.dzn")])
    if instance_path:
        file_entry_dzn.delete(0, tk.END)
        file_entry_dzn.insert(0, instance_path)
    
    #redundant fetch again:
    instance_path = file_entry_dzn.get()
    if not instance_path:
        messagebox.showinfo("Error", "Please select an instance file.")
        return
    if not instance_path.endswith(".dzn"):  # Assuming .dzn is the instance file type
        messagebox.showinfo("Error", "The selected file must be a .dzn file.")
        return
    
    model_path = file_entry_mzn.get()
    if not model_path.endswith(".mzn"):
        messagebox.showinfo("Error", "The selected file must be a .mzn model file.")
        return

    model = minizinc.Model(model_path)
    ortools_solver = minizinc.Solver.lookup("com.google.ortools.sat")
    instance = minizinc.Instance(ortools_solver, model)
    instance.add_file(instance_path)

    result = instance.solve()
    
    # Save the solution to a file in the 'solved_instances' directory
    solution_filename = os.path.basename(instance_path).replace('.dzn', '_solution.txt')
    solution_path = os.path.join('solved_instances', solution_filename)

    if result.solution is not None:
        with open(solution_path, 'w') as file:
            file.write(str(result))
        messagebox.showinfo("Solve Instances", f"Solution saved to {solution_path}")
    else:
        messagebox.showinfo("Solve Instances", "No solution found.")


def select_file():
    # Update to only accept .mzn files
    file_path = filedialog.askopenfilename(filetypes=[("MiniZinc files", "*.mzn")])
    if file_path:
        file_entry_mzn.delete(0, tk.END)
        file_entry_mzn.insert(0, file_path)
def analyze_data():
    # File selection
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        messagebox.showinfo("Error", "Please select a dataset.")
        return
    
    # Data handling
    data = pd.read_csv(file_path)
    data['subset_exists'] = data['subset_exists'].astype('category')
    train_data, test_data = train_test_split(data, test_size=0.33, random_state=1148, stratify=data['subset_exists'])

    # Models
    dt_classifier = DecisionTreeClassifier()
    xgb_classifier = XGBClassifier()
    rf_classifier = RandomForestClassifier(n_estimators=1000, max_features=2)

    # Training
    dt_classifier.fit(train_data.drop('subset_exists', axis=1), train_data['subset_exists'])
    xgb_classifier.fit(train_data.drop('subset_exists', axis=1), train_data['subset_exists'])
    rf_classifier.fit(train_data.drop('subset_exists', axis=1), train_data['subset_exists'])

    # Evaluation
    predictions_dt = dt_classifier.predict(test_data.drop('subset_exists', axis=1))
    predictions_xgb = xgb_classifier.predict(test_data.drop('subset_exists', axis=1))
    predictions_rf = rf_classifier.predict(test_data.drop('subset_exists', axis=1))

    # Store results in specified directory
    output_dir = 'analysis_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save ROC and Precision-Recall graphs
    evaluate_model(dt_classifier, test_data.drop('subset_exists', axis=1), test_data['subset_exists'], "Decision Tree", output_dir)
    evaluate_model(xgb_classifier, test_data.drop('subset_exists', axis=1), test_data['subset_exists'], "XGBoost", output_dir)
    evaluate_model(rf_classifier, test_data.drop('subset_exists', axis=1), test_data['subset_exists'], "Random Forest", output_dir)

    messagebox.showinfo("Analysis Complete", "Analysis and graphs saved in 'analysis_results' folder.")

def evaluate_model(model, test_features, test_labels, model_name, output_dir):
    probabilities = model.predict_proba(test_features)[:,1]
    fpr, tpr, _ = roc_curve(test_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'ROC_{model_name}.png'))
    plt.close()

    precision, recall, _ = precision_recall_curve(test_labels, probabilities)
    average_precision = average_precision_score(test_labels, probabilities)
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {average_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f'PR_{model_name}.png'))
    plt.close()

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
select_button = tk.Button(root, text="Select mzn constraint File", command=select_file)
select_button.pack(padx=10, pady=5)


# Create buttons for generating and solving instances
generate_button = tk.Button(root, text="Generate Instances", command=generate_instances)
generate_button.pack(padx=10, pady=5)

solve_button = tk.Button(root, text="Solve Instances", command=solve_instance)
solve_button.pack(padx=10, pady=5)

# Create a dropdown menu for model selection
dropdown1 = tk.OptionMenu(root, dropdown_var1, *options1)
dropdown1.pack(padx=10, pady=10)

analyze_button = tk.Button(root, text="Analyze Data", command=analyze_data)
analyze_button.pack(padx=10, pady=5)
root.mainloop()
