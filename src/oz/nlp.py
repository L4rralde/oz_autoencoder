import string
import re

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


def preprocess_str(text: str, language: str = "english") -> None:
    stemmer = SnowballStemmer(language)
    try:
        stop_words = set(stopwords.words(language))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words(language))
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[\d]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = text.split()
    # Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) >= 2]
    return ' '.join(tokens)


class BagOfWords:
    def __init__(self, **kwargs) -> None:
        self.vectorizer = CountVectorizer(lowercase=False, **kwargs)

    def __getattr__(self, key: str) -> object:
        return getattr(self.vectorizer, key)

    def __get_transform_generator(self, books: list) -> None:
        if type(books[0]) == str:
            return (
                preprocess_str(line)
                for line in books
            )
        try:
            return (
                preprocess_str(line)
                for book in books
                for line in book.readlines()
            )
        except:
            return (
                preprocess_str(line)
                for line in books.readlines()
            )

    def fit(self, books: list):
        generator = self.__get_transform_generator(books)
        self.vectorizer.fit(generator)
        return self

    def fit_transform(self, books: list) -> np.ndarray:
        generator = self.__get_transform_generator(books)
        result = self.vectorizer.fit_transform(generator)
        return result

    def transform(self, books):
        generator = self.__get_transform_generator(books)
        result = self.vectorizer.transform(generator)
        return result
