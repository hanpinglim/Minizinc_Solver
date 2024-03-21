import tkinter as tk
from tkinter import filedialog, messagebox

def generate_instances():
    # Your code to generate instances
    messagebox.showinfo("Generate Instances", "Instances generated successfully!")

def solve_instances():
    # Your code to solve instances
    messagebox.showinfo("Solve Instances", "Instances solved successfully!")

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

root = tk.Tk()
root.title("MiniZinc Instance Solver")

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

root.mainloop()