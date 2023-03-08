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
    print(find_division(data, 1))

def mse(y_true, y_pred):
    pass

def find_division(data: np.ndarray, col: int):
    sorted_indexes = np.argsort(data[:, col])
    return data[sorted_indexes]

predict(1)



class Node(object):
    def __init__(self):
        self.left = None  # Typ: Node, wierzchoĹek znajdujÄcy siÄ po lewej stornie
        self.right = None  # Typ: Node, wierzchoĹek znajdujÄcy siÄ po prawej stornie

    def perform_split(self, data):
        pass
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
        pass
        """
        if not Node jest liĹciem:
            if warunek podziaĹu jest speĹniony:
                return self.right.predict(example)
            else:
                return self.left.predict(example)
        return zwrĂłÄ wartoĹÄ (Node jest liĹciem)
        """

def Tree():
    pass

# Najprostsze wczytywanie i zapisywanie danych, a takĹźe wywoĹanie obiektu Node
# SzczegĂłlnie przydatne dla osĂłb nie znajÄcych numpy
# ZbiĂłr danych jest reprezentowany jako "lista list".
# tj. data[0] zwrĂłci listÄ z wartoĹciami cech pierwszego przykĹadu
# JeĹli znasz numpy i bolÄ CiÄ od poniĹźszego kodu oczy - moĹźesz go zmieniÄ
# JeĹli nie znasz numpy - skorzystaj z kodu, dokoĹcz zadanie... i naucz sie numpy. W kolejnych tygodniach bÄdziemy z niego korzystaÄ.

# id = "1"  # podaj id zbioru danych ktĂłry chcesz przetworzyÄ np. 1
# data = []
# path = "./data/"
# y = [line.strip() for line in open(path + id + '-Y.csv')]
# for i, line in enumerate(open(path + id + '-X.csv')):
#     if i == 0:
#         continue
#     x = [float(j) for j in line.strip().split(',')]
#     nAttr = len(x)
#     x.append(float(y[i]))
#     data.append(x)
#
# print(data)

print('Data load complete!')
# tree_root = Node()
# tree_root.perform_split(data)
# print('Training complete!')
#
# with open(id + '.csv', 'w') as f:
#     for i, line in enumerate(open(id + '-test.csv')):
#         if i == 0:
#             continue
#         x = [float(j) for j in line.strip().split(',')]
#         y = tree.predict(x)
#         f.write(str(y))
#         f.write('\n')