from preprocessing.get_data import process_train_test_data
from models.prepare_data import prepare_training_data

if __name__ == "__main__":
    train, _ = process_train_test_data(save=True)
    
    train_set, val_set = prepare_training_data(train, save=True)