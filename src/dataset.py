import os
from random import sample, shuffle

from tqdm import tqdm
import numpy as np
from scipy import sparse

from src.utils import GIT_ROOT
from src.books import CategoricalAuthors


MATRICES_PATH = f"{GIT_ROOT}/data/matrices"
X_PATH = f"{MATRICES_PATH}/sparse_X.npz"
Y_PATH = f"{MATRICES_PATH}/sparse_y.npz"
CATS_PATH = f"{MATRICES_PATH}/categories.cat"


class Dataset:
    def __init__(self, bow: object, books: object) -> None:
        self.bow = bow
        self.books = books
        self.dataset = ()

    def prepare(self, categoy: object, **params) -> None:
        self.bow.set_params(**params)
        print("Fitting Bag of words. This may take a while")
        self.bow.fit(self.books)

        print("Transforming each book")
        x_books = []
        t_books = []
        for book in tqdm(self.books):
            x_book = self.bow.transform(book)
            x_books.append(x_book)
            t_books.append(categoy(book.author)*np.ones(x_book.shape[0]))

        print("Appending sparse matrices")
        X = x_books[0].toarray()
        for i in tqdm(range(1, len(x_books))):
            X = np.concatenate([X, x_books[i].toarray()])
        y = np.concatenate(t_books)

        empty_lines = np.where(((X > 0).sum(axis=1) == 0))

        X_no_empty = np.delete(X, empty_lines, axis=0)
        y_no_empty = np.delete(y, empty_lines)

        print("Making the dataset sparse again")
        sparse_x = sparse.csr_matrix(X_no_empty)
        sparse_y = sparse.csr_matrix(y_no_empty)

        self.dataset = (sparse_x, sparse_y)
        self.to_memory()
        categoy.save(CATS_PATH)

    def to_memory(self) -> None:
        print(f"Saving checkpoint in memory")
        os.makedirs(MATRICES_PATH, exist_ok=True)
        sparse_x, sparse_y = self.dataset
        sparse.save_npz(X_PATH, sparse_x)
        sparse.save_npz(Y_PATH, sparse_y)

    @staticmethod
    def load_dataset(load_categories: bool = False) -> tuple:
        print("Loading checkpoint from memory")
        sparse_x = sparse.load_npz(X_PATH)
        sparse_y = sparse.load_npz(Y_PATH)
        x = sparse_x.toarray()
        y = sparse_y.toarray().flatten()
        if load_categories:
            return x, y, Dataset.load_categories()
        return x, y

    @staticmethod
    def load_categories() -> object:
        return CategoricalAuthors.from_file(CATS_PATH).to_dict()


def load_grouped_dataset(n: int) -> tuple:
    X, y, labels = Dataset.load_dataset(load_categories=True)
    X, y = group_obs(X, y, labels, n)
    return X, y, labels


def sub_dataset(*args, n: int = None) -> tuple:
    total = len(args[0])
    if n is None:
        n = total//2
    idcs = sample(range(total), n)
    return [arg[idcs] for arg in args]


def group_obs(X: np.ndarray, y: np.ndarray, labels: dict, m: int = 100) -> tuple:
    label_idcs = {
        label: [i for i, x in enumerate(y == label) if x]
        for label in labels.keys()
    }

    grouped_obs = {}
    for label in label_idcs.keys():
        label_obs = X[label_idcs[label]]
        M, N = label_obs.shape

        remainder = M % m
        if remainder != 0:
            pad_rows = m - remainder
            padding = np.zeros((pad_rows, N))
            label_obs = np.vstack([label_obs, padding])
        else:
            label_obs = label_obs

        M, N = label_obs.shape
        grouped = label_obs.reshape(M//m, m, N).sum(axis=1)
        grouped_obs[label] = grouped

    X_merged = np.vstack([*grouped_obs.values()])
    y_merged = np.zeros(X_merged.shape[0])

    start_idx = 0
    for label in label_idcs.keys():
        end_idx = start_idx + len(grouped_obs[label])
        y_merged[start_idx: end_idx] = label
        start_idx = end_idx
    return X_merged, y_merged


def shuffle_dataset(*args) -> None:
    idcs = list(range(len(args[0])))
    shuffle(idcs)
    return [
        [arg[i] for i in idcs]
        for arg in args
    ]


def load_twits_dataset() -> tuple:
    x_path = f"{MATRICES_PATH}/sparse_twits.npz"
    y_path = f"{MATRICES_PATH}/twits_labels.npz"
    sparse_x = sparse.load_npz(x_path)
    y = np.load(y_path)['arr_0']
    x = sparse_x.toarray()
    return x, y


