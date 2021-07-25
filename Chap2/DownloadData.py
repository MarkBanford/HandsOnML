import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

housing = pd.read_csv('housing.csv')
housing.hist(bins=50, figsize=(20, 15))


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_test_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_test_size]
    train_indices = shuffled_indices[test_test_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == '__main__':
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) # this does same as above
    print(len(train_set))
    print(len(test_set))
