import minizinc

# Load the Sudoku model
model = minizinc.Model("./mznc2023_probs/sudoku_fixed/sudoku_fixed.mzn") #Minizinc_Solver\mznc2023_probs\sudoku_fixed\sudoku_fixed.mzn

# Create a MiniZinc solver instance
gecode_solver = minizinc.Solver.lookup("gecode")

# Create an instance of the model and link the data file
instance = minizinc.Instance(gecode_solver, model)
instance.add_file("./mznc2023_probs/sudoku_fixed/sudoku_p19.dzn")

# Solve the problem
result = instance.solve()

# Print the solution
print(result)

