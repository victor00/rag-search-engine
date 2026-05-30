#!/usr/bin/env python3

import argparse
import json
import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer


BM25_K1 = 1.5

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


def single_tokenize(term: str, stopwords: set[str]) -> str:
    tokens = tokenize(term, stopwords)

    if len(tokens) != 1:
        raise ValueError(f"Expected a single token, got: {tokens}")

    return tokens[0]


class InvertedIndex:
    def __init__(self, stopwords: set[str]) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.stopwords = stopwords

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text, self.stopwords)
        self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        token = single_tokenize(term, self.stopwords)
        return sorted(self.index.get(token, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = single_tokenize(term, self.stopwords)
        return self.term_frequencies.get(doc_id, Counter()).get(token, 0)

    def get_bm25_tf(
        self,
        doc_id: int,
        term: str,
        k1: float = BM25_K1,
    ) -> float:
        tf = self.get_tf(doc_id, term)

        return (tf * (k1 + 1)) / (tf + k1)

    def get_idf(self, term: str) -> float:
        token = single_tokenize(term, self.stopwords)
        document_frequency = len(self.index.get(token, set()))

        if document_frequency == 0:
            return 0.0

        total_documents = len(self.docmap)
        return math.log(total_documents / document_frequency)

    def get_tfidf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_bm25_idf(self, term: str) -> float:
        token = single_tokenize(term, self.stopwords)
        document_frequency = len(self.index.get(token, set()))
        total_documents = len(self.docmap)

        return math.log(
            ((total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            + 1
        )

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

        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

    def load(self) -> None:
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)

        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)


def load_movies() -> list[dict]:
    with open("data/movies.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["movies"]


def load_index(stopwords: set[str]) -> InvertedIndex | None:
    inverted_index = InvertedIndex(stopwords)

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Error: index cache not found. Run the build command first.")
        return None

    return inverted_index


def bm25_idf_command(term: str) -> float:
    stopwords = load_stopwords()
    inverted_index = load_index(stopwords)

    if inverted_index is None:
        raise FileNotFoundError("Index cache not found. Run the build command first.")

    token = single_tokenize(term, stopwords)

    return inverted_index.get_bm25_idf(token)


def bm25_tf_command(
    doc_id: int,
    term: str,
    k1: float = BM25_K1,
) -> float:
    stopwords = load_stopwords()
    inverted_index = load_index(stopwords)

    if inverted_index is None:
        raise FileNotFoundError("Index cache not found. Run the build command first.")

    return inverted_index.get_bm25_tf(doc_id, term, k1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Single search term")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Single search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Single search term")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf",
        help="Get BM25 IDF score for a given term",
    )
    bm25_idf_parser.add_argument(
        "term",
        type=str,
        help="Term to get BM25 IDF score for",
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf",
        help="Get BM25 TF score for a given document ID and term",
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=BM25_K1,
        help="Tunable BM25 K1 parameter",
    )

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
            inverted_index = load_index(stopwords)

            if inverted_index is None:
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

        case "tf":
            stopwords = load_stopwords()
            inverted_index = load_index(stopwords)

            if inverted_index is None:
                return

            try:
                print(inverted_index.get_tf(args.doc_id, args.term))
            except ValueError as error:
                print(f"Error: {error}")

        case "idf":
            stopwords = load_stopwords()
            inverted_index = load_index(stopwords)

            if inverted_index is None:
                return

            try:
                idf = inverted_index.get_idf(args.term)
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            except ValueError as error:
                print(f"Error: {error}")

        case "tfidf":
            stopwords = load_stopwords()
            inverted_index = load_index(stopwords)

            if inverted_index is None:
                return

            try:
                tf_idf = inverted_index.get_tfidf(args.doc_id, args.term)
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            except ValueError as error:
                print(f"Error: {error}")

        case "bm25idf":
            try:
                bm25idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
            except (ValueError, FileNotFoundError) as error:
                print(f"Error: {error}")

        case "bm25tf":
            try:
                bm25tf = bm25_tf_command(
                    args.doc_id,
                    args.term,
                    args.k1,
                )
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
            except (ValueError, FileNotFoundError) as error:
                print(f"Error: {error}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()