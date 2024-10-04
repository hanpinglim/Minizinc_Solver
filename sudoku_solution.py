import pandas as pd
import os
from tkinter import Toplevel, Label, StringVar, OptionMenu, Entry, Button, messagebox
import random
import shutil
from tkinter import ttk



# Function to convert hex string to decimal
def convert_hex_to_decimal(hex_string, n):
    if n == 16:
        hex_map = {'A': '10', 'B': '11', 'C': '12', 'D': '13', 'E': '14', 'F': '15', 'G': '16'}
        return [hex_map.get(c, c) for c in hex_string]
    else:
        # No need to convert hex for 9x9, just return the values directly
        return list(hex_string)

# Function to count unknowns (zeros) in the puzzle/solution
def count_unknowns(instance):
    return instance.count('0')

# Function to adjust the number of unknowns based on user input
def adjust_unknowns(solution, current_problem, desired_unknowns):
    problem_list = list(current_problem)
    solution_list = list(solution)
    
    unknown_positions = [i for i, value in enumerate(problem_list) if value == '0']
    existing_unknowns = len(unknown_positions)
    
    if existing_unknowns <= desired_unknowns:
        return problem_list
    
    to_fill = existing_unknowns - desired_unknowns
    positions_to_fill = random.sample(unknown_positions, to_fill)
    for pos in positions_to_fill:
        problem_list[pos] = solution_list[pos]

    return problem_list

# Function to create the DZN file for each solution/problem
def save_sudoku_instance(instance, index, folder_path, n, instance_type, unknowns=None, solution=None):
    instance_decimal = convert_hex_to_decimal(instance, n)

    if unknowns and solution:
        solution_decimal = convert_hex_to_decimal(solution, n)
        instance_decimal = adjust_unknowns(solution_decimal, instance_decimal, unknowns)

    actual_unknowns = count_unknowns(instance_decimal)
    instance_grid = ', '.join(', '.join(instance_decimal[i:i+n]) for i in range(0, len(instance_decimal), n))

    dzn_content = f"""int: n = {n};
int: unknowns = {actual_unknowns};
array[1..n, 1..n] of var 0..n: x = [
  {instance_grid}
];
"""
    file_name = f"sudoku_p{index}.dzn"
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, 'w') as file:
        file.write(dzn_content)

# Function to fetch Sudoku solutions and save them in the chosen folders
def extract_solutions_and_problems(csv_file_path, num_solutions, output_solution_folder, output_problem_folder, n, fixed_unknowns=None, progress_bar=None):
    df = pd.read_csv(csv_file_path, nrows=100000)

    valid_solutions = []  
    valid_problems = []   
    
    for i in range(len(df)):
        problem = df['puzzle'][i]
        current_unknowns = count_unknowns(convert_hex_to_decimal(problem, n))

        if current_unknowns >= fixed_unknowns:
            valid_solutions.append(df['solution'][i])
            valid_problems.append(problem)

    if len(valid_solutions) < num_solutions:
        messagebox.showerror("Error", f"Only {len(valid_solutions)} puzzles meet the minimum unknown requirement of {fixed_unknowns}.")
        return

    selected_indices = random.sample(range(len(valid_solutions)), num_solutions)

    # Set the max value for the progress bar
    if progress_bar:
        progress_bar["maximum"] = num_solutions
        progress_bar["value"] = 0

    for i, idx in enumerate(selected_indices):
        solution = valid_solutions[idx]
        problem = valid_problems[idx]

        save_sudoku_instance(solution, i, output_solution_folder, n, "solution")
        save_sudoku_instance(problem, i, output_problem_folder, n, "problem")

        if fixed_unknowns is not None:
            output_fixed_folder = os.path.join(output_solution_folder, "..", "problems_fixed_unknowns")
            if not os.path.exists(output_fixed_folder):
                os.makedirs(output_fixed_folder)

            save_sudoku_instance(problem, i, output_fixed_folder, n, "problem", unknowns=fixed_unknowns, solution=solution)

        # Update the progress bar after each instance
        if progress_bar:
            progress_bar["value"] += 1
            progress_bar.update()

# Function to split instances into train and test folders based on percentage
def split_into_train_test(folder_path, percentage):
    problem_folder = os.path.join(folder_path, "problems")
    fixed_unknowns_folder = os.path.join(folder_path, "problems_fixed_unknowns")
    solution_folder = os.path.join(folder_path, "solutions")
    
    # Create train and test folders
    train_folder = os.path.join(folder_path, "train")
    test_folder = os.path.join(folder_path, "test")

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Create subfolders inside train and test
    train_problem_folder = os.path.join(train_folder, "problems")
    test_problem_folder = os.path.join(test_folder, "problems")
    os.makedirs(train_problem_folder, exist_ok=True)
    os.makedirs(test_problem_folder, exist_ok=True)

    train_fixed_folder = os.path.join(train_folder, "problems_fixed_unknowns")
    test_fixed_folder = os.path.join(test_folder, "problems_fixed_unknowns")
    os.makedirs(train_fixed_folder, exist_ok=True)
    os.makedirs(test_fixed_folder, exist_ok=True)

    train_solution_folder = os.path.join(train_folder, "solutions")
    test_solution_folder = os.path.join(test_folder, "solutions")
    os.makedirs(train_solution_folder, exist_ok=True)
    os.makedirs(test_solution_folder, exist_ok=True)

    # Function to split files between train and test folders
    def move_files(src_folder, train_dest, test_dest):
        files = os.listdir(src_folder)
        total_files = len(files)
        train_count = int(total_files * (percentage / 100))
        test_count = total_files - train_count

        # Move the first 'train_count' files to train and the rest to test
        for i, file_name in enumerate(files):
            src_file = os.path.join(src_folder, file_name)
            if i < train_count:
                shutil.move(src_file, os.path.join(train_dest, file_name))
            else:
                shutil.move(src_file, os.path.join(test_dest, file_name))

    # Move files in problems, problems_fixed_unknowns, and solutions
    move_files(problem_folder, train_problem_folder, test_problem_folder)
    move_files(fixed_unknowns_folder, train_fixed_folder, test_fixed_folder)
    move_files(solution_folder, train_solution_folder, test_solution_folder)

    # Remove the now-empty root problem and solution folders
    shutil.rmtree(problem_folder)
    shutil.rmtree(fixed_unknowns_folder)
    shutil.rmtree(solution_folder)


# Function to handle GUI prompts and actions
def run_extraction_gui():
    sudoku_window = Toplevel()
    sudoku_window.title("Sudoku Solution & Problem Extractor")
    
    n_var = StringVar(sudoku_window)
    n_var.set("9")
    
    Label(sudoku_window, text="Select grid size (n):").grid(row=0, column=0, padx=10, pady=10)
    grid_size_menu = OptionMenu(sudoku_window, n_var, "9", "16")
    grid_size_menu.grid(row=0, column=1, padx=10, pady=10)
    
    Label(sudoku_window, text="Number of solutions to extract:").grid(row=1, column=0, padx=10, pady=10)
    num_solutions_entry = Entry(sudoku_window)
    num_solutions_entry.grid(row=1, column=1, padx=10, pady=10)

    Label(sudoku_window, text="Number of unknowns:").grid(row=2, column=0, padx=10, pady=10)
    unknowns_entry = Entry(sudoku_window)
    unknowns_entry.grid(row=2, column=1, padx=10, pady=10)

    # Add new percentage input field
    Label(sudoku_window, text="Percentage of instances to train:").grid(row=3, column=0, padx=10, pady=10)
    percentage_entry = Entry(sudoku_window)
    percentage_entry.grid(row=3, column=1, padx=10, pady=10)

    # Add a progress bar
    progress_bar = ttk.Progressbar(sudoku_window, orient="horizontal", mode="determinate", length=200)
    progress_bar.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def on_submit():
        n = int(n_var.get())
        limit = 3000 if n == 16 else float('inf')
        unknowns_cap = 150 if n == 16 else 40 
        
        try:
            num_solutions = int(num_solutions_entry.get())
            if num_solutions > limit:
                messagebox.showerror("Error", f"Maximum available solutions for n={n} is {limit}!")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of solutions!")
            return

        try:
            desired_unknowns = int(unknowns_entry.get())
            if desired_unknowns > unknowns_cap:
                messagebox.showerror("Error", f"Maximum allowed unknowns for n={n} is {unknowns_cap}!")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of unknowns!")
            return

        # Fetch and validate the percentage
        try:
            percentage_value = float(percentage_entry.get())
            if not 0 <= percentage_value <= 100:
                messagebox.showerror("Error", "Please enter a percentage between 0 and 100!")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid percentage!")
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))

        if n == 9:
            csv_file_path = os.path.join(base_dir, "sudoku9x9", "sudoku.csv")
        elif n == 16:
            csv_file_path = os.path.join(base_dir, "sudoku16x16", "sudoku.csv")

        if not os.path.exists(csv_file_path):
            messagebox.showerror("Error", f"CSV file not found at {csv_file_path}!")
            return

        output_solution_folder = os.path.join(base_dir, "solved_instances", "solutions")
        output_problem_folder = os.path.join(base_dir, "solved_instances", "problems")

        if not os.path.exists(output_solution_folder):
            os.makedirs(output_solution_folder)
        if not os.path.exists(output_problem_folder):
            os.makedirs(output_problem_folder)

        try:
            # Extract the solutions and problems
            extract_solutions_and_problems(csv_file_path, num_solutions, output_solution_folder, output_problem_folder, n, fixed_unknowns=desired_unknowns, progress_bar=progress_bar)

            # After extraction, split the instances into train and test folders
            split_into_train_test(os.path.join(base_dir, "solved_instances"), percentage_value)
            
            # Show success message after splitting
            messagebox.showinfo("Success", f"Extracted {num_solutions} solutions and problems with {desired_unknowns} unknowns!\nTraining percentage: {percentage_value}%")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    submit_button = Button(sudoku_window, text="Submit", command=on_submit)
    submit_button.grid(row=5, columnspan=2, padx=10, pady=20)