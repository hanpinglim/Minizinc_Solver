import minizinc

# Load the Sudoku model
model = minizinc.Model("./mznc2023_probs/kidney-exchange/ccmcp.mzn") #Minizinc_Solver\mznc2023_probs\sudoku_fixed\sudoku_fixed.mzn

# Create a MiniZinc solver instance
gecode_solver = minizinc.Solver.lookup("gecode")

# Create an instance of the model and link the data file
instance = minizinc.Instance(gecode_solver, model)
instance.add_file("./mznc2023_probs/kidney-exchange/3_20_0.15_3.dzn")

# Solve the problem
result = instance.solve()
analyse_input = instance.input
analyse_method = instance.method
analyse_output = instance.output
# Print the solution
print(result)

