import random

def generate_instance(V, K):
    edge_weight = '[|'
    for i in range(V):
        row = []
        for j in range(V):
            if i == j:
                edge_weight += '0,'
            else:
                weight = random.randint(-100, 100)
                edge_weight += str(weight) + ','
        edge_weight += '|'

    edge_weight += ']'

    instance = f"""
K = {K};
V = {V};
edge_weight = {edge_weight};
"""
    return instance

def save_instances(num_instances, V, K, filename):
    for i in range(num_instances):
        with open('4701/instances/'+str(i)+filename, "w") as f:
            instance = generate_instance(V, K)
            f.write(instance)
            f.write("\n")

# Example usage
num_instances = 1000
V = 25
K = 3
filename = "instance.dzn"

save_instances(num_instances, V, K, filename)
print(f"Generated {num_instances} instances and saved to {filename}")