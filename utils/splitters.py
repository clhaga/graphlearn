from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

def split_graphs(dataset):
    train_idx, test_idx = train_test_split(range(len(dataset)), stratify=[m.y[0].item() for m in dataset], test_size=0.2)
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    return train_dataset, test_dataset

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    return train_data, test_data