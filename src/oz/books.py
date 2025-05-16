import os
from io import TextIOWrapper

import urllib.request

from src.utils import GIT_ROOT


BOOKS_PATH = f"{GIT_ROOT}/data/books"


class Book:
    DIRNAME = ""
    def __init__(
            self,
            title: str,
            author: str,
            path: str
    ) -> None:
        self.title = title
        self.author = author.lower()
        self.path = path

    def __repr__(self) -> str:
        return f"{self.title}. {self.author}"

    def download(self, url: str) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        print(f"Downloading {self.title}")
        data = urllib.request.urlopen(url)
        with open(self.path, 'w') as f:
            f.write(data.read().decode('utf-8'))

    def readlines(self) -> None:
        with open(self.path, "r") as f:
            lines = f.read().splitlines()
        return lines



class UrlBook(Book):
    DIRNAME = f"{BOOKS_PATH}/urls/"
    def __init__(self, title: str, author: str, url) -> None:
        path = f"{UrlBook.DIRNAME}/{"_".join(title.split())}"
        super().__init__(title, author, path)
        self.url = url

    def download(self) -> None:
        return super().download(self.url)
    
    def read(self) -> None:
        if not os.path.exists(self.path):
            self.download()
        return super().read()


OZBOOKS = (
    UrlBook(
        "Captain Salt in Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/56073/pg56073.txt"
    ),
    UrlBook(
        "Dorothy and the Wizard in Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/420/pg420.txt"
    ),
    UrlBook(
        "The Emerald City of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/517/pg517.txt"   
    ),
    UrlBook(
        "Glinda of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/961/pg961.txt"
    ),
    UrlBook(
        "Grampa in Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/61681/pg61681.txt"
    ),
    UrlBook(
        "Handy Mandy in Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/56079/pg56079.txt"
    ),
    UrlBook(
        "Kabumpo in Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/53765/pg53765.txt"
    ),
    UrlBook(
        "The Lost Princess of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/959/pg959.txt"
    ),
    UrlBook(
        "The Magic of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/419/pg419.txt"
    ),
    UrlBook(
        "The Marvelous Land of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/54/pg54.txt"
    ),
    UrlBook(
        "Ozma of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/486/pg486.txt"
    ),
    UrlBook(
        "Ozoplaning with the Wizard of Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/55806/pg55806.txt"
    ),
    UrlBook(
        "The Patchwork Girl of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/955/pg955.txt"
    ),
    UrlBook(
        "Rinkitink in Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/958/pg958.txt"
    ),
    UrlBook(
        "The Road to Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/485/pg485.txt"
    ),
    UrlBook(
        "The Royal Book of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/30537/pg30537.txt"
    ),
    UrlBook(
        "The Scarecrow of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/957/pg957.txt"
    ),
    UrlBook(
        "The Cowardly Lion of Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/58765/pg58765.txt"
    ),
    UrlBook(
        "The Hungry Tiger of Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/70152/pg70152.txt"
    ),
    UrlBook(
        "The Lost King of Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/65849/pg65849.txt"
    ),
    UrlBook(
        "The Magical Mimics in Oz",
        "Jack Snow",
        "https://www.gutenberg.org/cache/epub/56555/pg56555.txt"
    ),
    UrlBook(
        "The Shaggy Man of Oz",
        "Jack Snow",
        "https://www.gutenberg.org/cache/epub/56683/pg56683.txt"
    ),
    UrlBook(
        "The Silver Princess in Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/56085/pg56085.txt"
    ),
    UrlBook(
        "The Tin Woodman of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/30852/pg30852.txt"
    ),
    UrlBook(
        "The Wishing Horse of Oz",
        "Ruth Plumly Thompson",
        "https://www.gutenberg.org/cache/epub/55851/pg55851.txt"
    ),
    UrlBook(
        "Tik-Tok of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/52176/pg52176.txt"
    ),
    UrlBook(
        "The Woggle-Bug Book",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/21914/pg21914.txt"
    ),
    UrlBook(
        "The Wonderful Wizard of Oz",
        "L. Frank Baum",
        "https://www.gutenberg.org/cache/epub/55/pg55.txt"
    )
)


class CategoricalAuthors:
    def __init__(self) -> None:
        self.authors = []

    def fit(self, books: list) -> None:
        self.authors = list(set((book.author.lower() for book in books)))

    def transform(self, author: str) -> int:
        return self.authors.index(author.lower())

    def __call__(self, author: str) -> int:
        return self.transform(author)
