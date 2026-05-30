#!/usr/bin/env python3

import argparse
import json
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


def matches(query: str, title: str, stopwords: set[str]) -> bool:
    query_tokens = tokenize(query, stopwords)
    title_tokens = tokenize(title, stopwords)

    return any(
        query_token in title_token
        for query_token in query_tokens
        for title_token in title_tokens
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            with open("data/movies.json", "r", encoding="utf-8") as file:
                data = json.load(file)

            stopwords = load_stopwords()
            results = []

            for movie in data["movies"]:
                if matches(args.query, movie["title"], stopwords):
                    results.append(movie)

            print(f"Searching for: {args.query}")

            for index, movie in enumerate(results[:5], start=1):
                print(f"{index}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()