#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import string

from nltk.stem import PorterStemmer


stemmer = PorterStemmer()


def normalize(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator)


def load_stopwords() -> set[str]:
    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        words = file.read().splitlines()

    return {normalize(word) for word in words if word}


def tokenize(text: str, stopwords: set[str]) -> list[str]:
    return [
        stemmer.stem(token)
        for token in normalize(text).split()
        if token and token not in stopwords
    ]


class InvertedIndex:
    def __init__(self, stopwords: set[str]) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.stopwords = stopwords

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text, self.stopwords)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        tokens = tokenize(term.lower(), self.stopwords)

        if not tokens:
            return []

        return sorted(self.index.get(tokens[0], set()))

    def build(self, movies: list[dict]) -> None:
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(
                doc_id,
                f"{movie['title']} {movie['description']}",
            )

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)

        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

    def load(self) -> None:
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)


def load_movies() -> list[dict]:
    with open("data/movies.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["movies"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build and cache the inverted index")

    args = parser.parse_args()

    match args.command:
        case "build":
            movies = load_movies()
            stopwords = load_stopwords()

            inverted_index = InvertedIndex(stopwords)
            inverted_index.build(movies)
            inverted_index.save()

            print("Index built and saved.")

        case "search":
            stopwords = load_stopwords()
            inverted_index = InvertedIndex(stopwords)

            try:
                inverted_index.load()
            except FileNotFoundError:
                print("Error: index cache not found. Run the build command first.")
                return

            print(f"Searching for: {args.query}")

            results: list[int] = []

            for token in tokenize(args.query, stopwords):
                for doc_id in inverted_index.get_documents(token):
                    if doc_id not in results:
                        results.append(doc_id)

                    if len(results) >= 5:
                        break

                if len(results) >= 5:
                    break

            for index, doc_id in enumerate(results, start=1):
                movie = inverted_index.docmap[doc_id]
                print(f"{index}. {movie['title']} ({doc_id})")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()