import tkinter as tk
from tkinter import ttk

# Function to be called when the first button is clicked
def button1_clicked():
    print(f"Dropdown 1: {dropdown_var1.get()}, Dropdown 2: {dropdown_var2.get()}")

# Function to be called when the second button is clicked
def button2_clicked():
    print(f"Dropdown 2: {dropdown_var2.get()}, Dropdown 1: {dropdown_var1.get()}")

# Create the main window
root = tk.Tk()
root.title("Tkinter Dropdown Example")

# Define tkinter variable for dropdown selection
dropdown_var1 = tk.StringVar(root)
dropdown_var2 = tk.StringVar(root)

# Set default value for the dropdown
dropdown_var1.set("Option 1")
dropdown_var2.set("Option A")

# Options for dropdown menus
options1 = ["Option 1", "Option 2", "Option 3"]
options2 = ["Option A", "Option B", "Option C"]

# Create dropdown menus
dropdown1 = ttk.OptionMenu(root, dropdown_var1, *options1)
dropdown2 = ttk.OptionMenu(root, dropdown_var2, *options2)

# Create buttons
button1 = ttk.Button(root, text="Print Dropdowns 1 & 2", command=button1_clicked)
button2 = ttk.Button(root, text="Print Dropdowns 2 & 1", command=button2_clicked)

# Arrange the widgets using grid layout
dropdown1.grid(row=0, column=0, padx=10, pady=10)
dropdown2.grid(row=0, column=1, padx=10, pady=10)
button1.grid(row=1, column=0, padx=10, pady=10)
button2.grid(row=1, column=1, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
