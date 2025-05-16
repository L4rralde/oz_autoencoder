import os

from src.books import OZBOOKS, CategoricalAuthors
from src.nlp import BagOfWords
from src.dataset import Dataset, X_PATH, Y_PATH


def make():
    if os.path.exists(X_PATH) and os.path.exists(Y_PATH):
        print("Already in memory. Aborting!")
        return

    bow = BagOfWords(max_features=1000)

    cat = CategoricalAuthors()
    cat.fit(OZBOOKS)
    ds = Dataset(bow, OZBOOKS)
    ds.prepare(cat)
