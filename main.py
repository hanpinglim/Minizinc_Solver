import minizinc

# Load the Sudoku model
model = minizinc.Model("./sudoku_model.mzn")

# Create a MiniZinc solver instance
gecode_solver = minizinc.Solver.lookup("gecode")

# Create an instance of the model and link the data file
instance = minizinc.Instance(gecode_solver, model)
instance.add_file("./sudoku_p0.dzn")

# Solve the instance
result = instance.solve()

# Check and print the result
if result.status == minizinc.result.Status.SATISFIED:
    print("Solution found:")
    print(result.solution)
else:
    print("No solution found:", result.status)
