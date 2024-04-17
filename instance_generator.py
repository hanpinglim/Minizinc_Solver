import re
import random

def parse_dzn_file(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        content = file.read()
        matches = re.findall(r'(\w+)\s*=\s*([^;]+);', content)
        for name, value in matches:
            if '[' in value:  # This is an array or matrix
                # Further processing to understand array/matrix dimensions
                parameters[name] = {'type': 'matrix', 'value': eval(value)}
            else:
                try:
                    parameters[name] = {'type': 'int', 'value': int(value)}
                except ValueError:
                    continue  # Add other types as necessary
    return parameters

def generate_data(parameters):
    new_data = {}
    for name, info in parameters.items():
        if info['type'] == 'int':
            # Generate integers within some range
            new_data[name] = random.randint(1, 100)  # Example range
        elif info['type'] == 'matrix':
            # Generate matrix data respecting original dimensions and constraints
            matrix = info['value']
            rows = len(matrix)
            cols = len(matrix[0]) if rows else 0
            new_matrix = [[random.randint(-1, 100) for _ in range(cols)] for _ in range(rows)]
            new_data[name] = new_matrix
    return new_data

def create_dzn_content(data):
    content = ""
    for name, value in data.items():
        if isinstance(value, list):  # Assuming a matrix
            content += f"{name} = [|\n" + "\n".join(','.join(map(str, row)) + '|' for row in value) + "\n];\n"
        else:
            content += f"{name} = {value};\n"
    return content

# Example usage
params = parse_dzn_file('sudoku_p.dzn')
new_instance_data = generate_data(params)
new_dzn_content = create_dzn_content(new_instance_data)
print(new_dzn_content)
