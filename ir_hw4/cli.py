import argparse
import json
from pathlib import Path
from .config import RUN, METHODS


def main():
    parser = argparse.ArgumentParser(description="Independent Reuters HW4 experiments")
    parser.add_argument("--run", type=Path, default=RUN)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Download and validate the original Reuters SGML")
    sub.add_parser("train", help="Explicit fresh CBOW/Skip-Gram training; refuses overwrite")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--method", choices=METHODS, default="tfidf")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--output", type=Path)
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "prepare":
        from .dataset import prepare_data
        result = prepare_data()
    elif args.command == "train":
        from .training import train
        result = train(args.run)
    elif args.command == "verify":
        from .validation import verify
        result = verify(args.run)
    else:
        from .training import load_engine
        engine, _ = load_engine(args.run)
        if args.command == "search":
            ranking, status = engine.rank(args.query, args.method, 10)
            result = {"ranking": ranking, "query_status": status}
        else:
            from .evaluation import save_evaluation
            result = save_evaluation(engine, args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
