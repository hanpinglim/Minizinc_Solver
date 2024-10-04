import tkinter as tk
from sudoku_solution import run_extraction_gui as run_sudoku_extraction_gui
from kidney_exchange_solution import run_kidney_exchange_gui  
from training_gui import run_training_gui  # Import the training model GUI function

def open_sudoku_gui():
    """Function to run Sudoku Solver GUI."""
    # Call the Sudoku extraction GUI function
    run_sudoku_extraction_gui()

def open_kidney_exchange_gui():
    """Function to run Kidney Exchange Solver GUI."""
    # Call the Kidney Exchange GUI function
    run_kidney_exchange_gui()

def open_training_gui():
    """Function to run Training Model GUI."""
    # Call the Training Model GUI function
    run_training_gui()

def create_main_menu():
    """Main menu where the user selects the problem to solve."""
    root = tk.Tk()
    root.title("Select Problem to Solve")

    label = tk.Label(root, text="Choose a problem to solve:", font=("Arial", 14))
    label.pack(pady=20)

    # Sudoku Button
    sudoku_button = tk.Button(root, text="Sudoku", font=("Arial", 12), width=20, command=open_sudoku_gui)
    sudoku_button.pack(pady=10)

    # Kidney Exchange Button
    kidney_button = tk.Button(root, text="Kidney Exchange", font=("Arial", 12), width=20, command=open_kidney_exchange_gui)
    kidney_button.pack(pady=10)

    # Training Model Button
    training_button = tk.Button(root, text="Train a Model", font=("Arial", 12), width=20, command=open_training_gui)
    training_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_main_menu()
