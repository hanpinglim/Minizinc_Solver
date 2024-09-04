import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save to CSV
csv_path = 'iris_dataset.csv'
df.to_csv(csv_path, index=False)

print(f'Dataset saved to {csv_path}')

# Load the CSV file to verify
df_loaded = pd.read_csv(csv_path)
print(df_loaded.head())
