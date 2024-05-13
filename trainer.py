from kidney_exchange_dataset import KidneyExchangeDataset
import os

def process_data():
    # Get the base directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Correctly set the directory for processed data within 'solved_instances'
    processed_data_directory = os.path.join(base_dir, 'processed')

    # Create an instance of the dataset
    dataset = KidneyExchangeDataset(root=base_dir, processed_dir=processed_data_directory)
    try:
        dataset.process()
        print(f"Dataset length: {dataset.len()}")
        if dataset.len() > 0:
            sample_data = dataset.get(0)
            print(f"Sample data: {sample_data}")
        return True
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return False

if __name__ == "__main__":
    process_data()
