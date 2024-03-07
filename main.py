import minizinc

# Load the Sudoku model
model = minizinc.Model("./mznc2023_probs/sudoku_fixed/sudoku_fixed.mzn")

# Create a MiniZinc solver instance
gecode_solver = minizinc.Solver.lookup("gecode")

# Create an instance of the model and link the data file
instance = minizinc.Instance(gecode_solver, model)
instance.add_file("./sudoku_p0.dzn")

# Set the initial values of the Sudoku grid
instance["grid"] = [
    [0, 0, 0, 2, 0, 5, 0, 0, 0],
    [0, 9, 0, 0, 0, 0, 7, 3, 0],
    [0, 0, 2, 0, 0, 9, 0, 6, 0],
    [2, 0, 0, 0, 0, 0, 4, 0, 9],
    [0, 0, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 9, 0, 0, 0, 0, 0, 1],
    [0, 8, 0, 4, 0, 0, 1, 0, 0],
    [0, 6, 3, 0, 0, 0, 0, 8, 0],
    [0, 0, 0, 6, 0, 8, 0, 0, 0]
]
# Set the value of n
instance["n"] = 9
# Solve the problem
result = instance.solve()

# Print the solution
print(result)

