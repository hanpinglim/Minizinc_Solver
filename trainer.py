from kidney_exchange_dataset import KidneyExchangeDataset
import os

def process_data():
    # Get the base directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Process training data
    train_processed_data_directory = os.path.join(base_dir, 'processed', 'train')
    train_dataset = KidneyExchangeDataset(root=base_dir, processed_dir=train_processed_data_directory, train=True)
    try:
        train_dataset.process()
        print(f"Training dataset length: {train_dataset.len()}")
        if train_dataset.len() > 0:
            sample_train_data = train_dataset.get(0)
            print(f"Sample training data: {sample_train_data}")
    except Exception as e:
        print(f"An error occurred during processing training data: {e}")
        return False

    # Process testing data
    test_processed_data_directory = os.path.join(base_dir, 'processed', 'test')
    test_dataset = KidneyExchangeDataset(root=base_dir, processed_dir=test_processed_data_directory, train=False)
    try:
        test_dataset.process()
        print(f"Testing dataset length: {test_dataset.len()}")
        if test_dataset.len() > 0:
            sample_test_data = test_dataset.get(0)
            print(f"Sample testing data: {sample_test_data}")
        return True
    except Exception as e:
        print(f"An error occurred during processing testing data: {e}")
        return False

if __name__ == "__main__":
    process_data()
