#!/usr/bin/env python3

import argparse
import json
import string


def normalize(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.lower().translate(translator)


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

            query = normalize(args.query)
            results = []

            for movie in data["movies"]:
                title = normalize(movie["title"])

                if query in title:
                    results.append(movie)

            print(f"Searching for: {args.query}")

            for index, movie in enumerate(results[:5], start=1):
                print(f"{index}. {movie['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()