from multiprocessing.connection import deliver_challenge
import pandas as pd
import numpy as np

def read_data(path: str):
    return pd.read_csv(path)

def crossvalidation(data, k: int = 10):
    print(np.shape(data)[0])

    subset_size = np.shape(data)[0] // k

    for i in range(k):
        data[i * subset_size : (i+1) * subset_size]

def predict(no: int):
    X = read_data(f"./data/{no}-X.csv")
    y = read_data(f"./data/{no}-Y.csv")
    data = pd.concat([X, y], axis=1).values
    # crossvalidation(data, 10)
    split_mse, split_val, split_col = find_split(data)
    return split_val, split_col

def mse(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)

def find_split(data: np.ndarray):
    min_loss = None
    div_value = None
    min_col = None
    population = np.shape(data)[0]
    for col in range(np.shape(data)[1] - 1):
        sorted_indexes = np.argsort(data[:, col])
        sorted_data = data[sorted_indexes]
        _, unique_indices = np.unique(sorted_data[:, col], return_index=True)

        for index in unique_indices[1:]:
            left = sorted_data[:index]
            right = sorted_data[index:]

            loss = min(mse(np.mean(left[:, -1]), sorted_data[:, -1]), \
                mse(np.mean(right[:, -1]), sorted_data[:, -1]))

            if (min_loss == None or loss < min_loss) and np.shape(left)[0] / population > 0.1 and np.shape(right)[0] / population > 0.1:
                min_loss = loss
                div_value = sorted_data[index, col]
                min_col = col
    
    return min_loss, div_value, min_col

predict(1)



class Node():
    def __init__(self, data, level=0, max_level=5, is_leaf=False):
        self.left = None  # Typ: Node, wierzchoĹek znajdujÄcy siÄ po lewej stornie
        self.right = None  # Typ: Node, wierzchoĹek znajdujÄcy siÄ po prawej stornie
        self.is_leaf = is_leaf
        self.level = level
        self.max_level = max_level
        self.col = None
        self.value = None
        self.predict_value = 0
        self.data = data


    def perform_split(self):
        if self.level >= self.max_level:
            self.is_leaf = True
            self.predict_value = np.mean(self.data[:, -1])
            print(self.predict_value)
        else:
            _, self.value, self.col = find_split(self.data)
            print(self.value)
            print(self.col)
            # print(np.shape(self.data))
            condition = self.data[:, self.col] < self.value
            print(condition)
            left_data = self.data[condition]
            right_data = self.data[~condition]

            print(np.shape(left_data))
            print(np.shape(right_data))

            self.left = Node(left_data, self.level + 1, self.max_level)
            self.right = Node(right_data, self.level + 1, self.max_level)

            self.left.perform_split()
            self.right.perform_split()

    # ZnajdĹş najlepszy podziaĹ data
    # if uzyskano poprawÄ funkcji celu (bÄdĹş inny, zaproponowany przez Ciebie warunek):
    # podziel dane na dwie czÄĹci d1 i d2, zgodnie z warunkiem
    # self.left = Node()
    # self.right = Node()
    # self.left.perform_split(d1)
    # self.right.perform_split(d2)
    # else:
    # obecny Node jest liĹciem, zapisz jego odpowiedĹş

    def predict(self, example):
        if self.is_leaf == True:
            print("XDDDD")
            return self.predict_value
        else:
            if example[self.col] <= self.value:
                self.left.predict(example)
            else:
                self.right.predict(example)

# X = read_data(f"./data/1-X.csv")
# y = read_data(f"./data/1-Y.csv")
# data = pd.concat([X, y], axis=1).values

# tree = Node(data)
# tree.perform_split()