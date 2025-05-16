import os

from tqdm import tqdm
import numpy as np
from scipy import sparse

from src.utils import GIT_ROOT


MATRICES_PATH = f"{GIT_ROOT}/data/matrices"
X_PATH = f"{MATRICES_PATH}/sparse_X.npz"
Y_PATH = f"{MATRICES_PATH}/sparse_y.npz"

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

    def to_memory(self) -> None:
        print(f"Saving checkpoint in memory")
        os.makedirs(MATRICES_PATH, exist_ok=True)
        sparse_x, sparse_y = self.dataset
        sparse.save_npz(X_PATH, sparse_x)
        sparse.save_npz(Y_PATH, sparse_y)   

    @staticmethod
    def load_dataset() -> tuple:
        print("Loading checkpoint from memory")
        sparse_x = sparse.load_npz(X_PATH)
        sparse_y = sparse.load_npz(Y_PATH)
        return sparse_x.toarray(), sparse_y.toarray().flatten()
