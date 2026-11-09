"""User-configured offline Word2Vec runs built from the verified HW4 training inputs."""

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
from time import perf_counter

from gensim.models import Word2Vec
import numpy as np
from scipy import sparse

from .analysis import checked_evaluation
from .config import ANALYSIS_WORDS, DATA, METHODS, ROOT, RUN
from .evaluation import save_evaluation
from .io import read_json, sha256, stable_hash, write_csv, write_json
from .retrieval import neighbors, weighted_documents
from .training import EpochRecorder, PRODUCER_FILES, TrainingConfig, load_engine

MODEL_RUNS = ROOT / "artifacts" / "experiments"
EVALUATION_RUNS = ROOT / "reports" / "experiments"
PARAMETERS = {
    "vector_size": (1, 1024, "每個詞的向量維度"),
    "window": (1, 40, "同一句內中心詞與上下文詞的最大距離"),
    "min_count": (1, 10000, "訓練詞頻低於此值的詞不進入 Word2Vec 詞表"),
    "epochs": (1, 300, "遍歷訓練語料的次數"),
    "negative": (1, 40, "負採樣數量"),
    "sample": (0.0, 1.0, "高頻詞下採樣設定，0 表示關閉"),
    "seed": (0, 2**32 - 1, "隨機種子"),
    "workers": (1, 32, "訓練執行緒數，固定為 1 較容易重現"),
}
DEFAULT_CONFIG = TrainingConfig()
ARCHITECTURES = {"cbow": 0, "skipgram": 1}
SHARED_INPUTS = ("training_sentences.jsonl", "document_order.json", "test_tfidf.npz",
                 "term_statistics.npz", "vocabulary.json", "term_statistics.csv")
WORD_MODEL_FILES = ("word2vec.model", "vectors.npy", "vocabulary.json", "epoch_losses.csv",
                    "test_document_vectors.npy", "test_weight_coverage.npy", "metadata.json")
EXPERIMENT_SOURCE = "ir_hw4/experiments.py"
EVALUATION_FILES = {"results.json", "metrics_summary.csv", "metrics_per_query.csv",
                    "rankings.csv", "queries.json", "REPORT.md"}


def experiment_config(values):
    if values.keys() - PARAMETERS.keys():
        raise ValueError("Unknown experiment parameters")
    config = replace(DEFAULT_CONFIG, **values)
    for field, (lower, upper, _) in PARAMETERS.items():
        value = getattr(config, field)
        if type(value) is not type(getattr(DEFAULT_CONFIG, field)) or not lower <= value <= upper:
            raise ValueError(f"{field} must be between {lower} and {upper}")
    return config


def experiment_paths(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", name) or PureWindowsPath(name).is_reserved():
        raise ValueError("Experiment name must use 1–48 letters, numbers, hyphens or underscores")
    paths = MODEL_RUNS / name, EVALUATION_RUNS / name
    for path in paths:
        if path.resolve() != path.parent.resolve() / name or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Experiment path escapes its project directory")
    return paths


def completed_experiments():
    if not EVALUATION_RUNS.is_dir():
        return []
    return sorted(path.name for path in EVALUATION_RUNS.iterdir() if (path / "ready.json").is_file())


def check_hashes(directory, recorded, required):
    if not set(required).issubset(recorded):
        raise ValueError(f"Missing required fingerprints in {directory.name}")
    root = directory.resolve()
    for relative, expected in recorded.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Fingerprint mismatch: {relative}")


def train_experiment(name, settings):
    """Train both architectures, evaluate once, then make the completed run visible."""
    config = experiment_config(settings)
    run, output = experiment_paths(name)
    if run.exists() or output.exists():
        raise FileExistsError(f"Experiment {name} already exists. Choose a new name.")
    base_engine, base = load_engine(RUN)
    base_signature = sha256(RUN / "manifest.json")
    checked_evaluation(base_signature)
    if sha256(DATA / "manifest.json") != base["dataset_manifest_sha256"]:
        raise ValueError("The data manifest differs from the prepared training inputs")
    if np.count_nonzero(base_engine.space.cf >= config.min_count) < 2:
        raise ValueError("min_count leaves fewer than two training words")

    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "manifest.json", {"status": "training", "experiment_name": name, "config": asdict(config)})
    for filename in SHARED_INPUTS:
        shutil.copyfile(RUN / filename, run / filename)
    check_hashes(run, {filename: base["artifacts_sha256"][filename] for filename in SHARED_INPUTS}, SHARED_INPUTS)
    training_ids = set(read_json(run / "document_order.json")["train"])
    sentences = []
    with (run / "training_sentences.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] not in training_ids:
                raise ValueError("A training sentence came from outside the training set")
            sentences.append(row["tokens"])
    if len(sentences) != base["training_sentences"] or sum(map(len, sentences)) != base["training_tokens"]:
        raise ValueError("Prepared training sentence counts differ from the source run")

    lexical = sparse.load_npz(run / "test_tfidf.npz")
    metadata, word_models = {}, {}
    for model_name, sg in ARCHITECTURES.items():
        options = replace(config, sg=sg)
        print(f"{name} · {model_name} · {options.vector_size} dimensions · window {options.window}", flush=True)
        started = perf_counter()
        model = Word2Vec(**asdict(options), hashfxn=stable_hash, hs=0, batch_words=10000)
        model.build_vocab(sentences)
        initial_digest = hash_bytes(model.wv.vectors.tobytes()).hexdigest()
        callback = EpochRecorder(model_name)
        effective, raw = model.train(sentences, total_examples=model.corpus_count, epochs=options.epochs,
                                     compute_loss=True, callbacks=[callback])
        words, vectors = model.wv.index_to_key, model.wv.vectors
        if vectors.shape != (len(words), config.vector_size) or not np.isfinite(vectors).all() or len(callback.rows) != options.epochs:
            raise ValueError(f"Incomplete {model_name} training")
        target = run / model_name
        target.mkdir()
        model.save(str(target / "word2vec.model"), separately=[])
        reloaded = Word2Vec.load(str(target / "word2vec.model"))
        if words != reloaded.wv.index_to_key or not np.array_equal(vectors, reloaded.wv.vectors):
            raise ValueError(f"{model_name} reload mismatch")
        np.save(target / "vectors.npy", vectors, allow_pickle=False)
        write_json(target / "vocabulary.json", words)
        write_csv(target / "epoch_losses.csv", callback.rows)
        averaged, coverage = weighted_documents(lexical, base_engine.space.word_matrix(words, vectors))
        np.save(target / "test_document_vectors.npy", averaged, allow_pickle=False)
        np.save(target / "test_weight_coverage.npy", coverage, allow_pickle=False)
        metadata[model_name] = {
            "architecture": "Skip-Gram" if sg else "CBOW", "config": asdict(options),
            "vocabulary": len(words), "seconds": perf_counter() - started,
            "initial_vectors_sha256": initial_digest, "offline_reload_exact": True,
            "train_documents": len(training_ids), "test_documents_used_for_training": 0,
            "effective_training_words_all_epochs": int(effective), "raw_training_words_all_epochs": int(raw),
            "initialization": "Fresh Word2Vec trained on the verified train-only sentence sequences.",
        }
        write_json(target / "metadata.json", metadata[model_name])
        word_models[model_name] = (words, vectors)

    with (run / "word_neighbors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("anchor", "model", "rank", "word", "cosine"))
        writer.writeheader()
        for word in ANALYSIS_WORDS:
            for model_name, model in word_models.items():
                for rank, row in enumerate(neighbors(*model, word), 1):
                    writer.writerow({"anchor": word, "model": model_name, "rank": rank, **row})

    manifest = {
        **base, "status": "complete", "experiment_name": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_manifest_sha256": base_signature,
        "embedding": f"Shared TF-IDF weights with new Word2Vec vectors. Cosine in {config.vector_size} dimensions.",
        "models": metadata,
        "source_sha256": {**base["source_sha256"], EXPERIMENT_SOURCE: sha256(Path(__file__))},
        "artifacts_sha256": {p.relative_to(run).as_posix(): sha256(p) for p in sorted(run.rglob("*"))
                             if p.is_file() and p.name != "manifest.json"},
    }
    write_json(run / "manifest.json", manifest)
    engine, _ = load_engine(run)
    summary = save_evaluation(engine, run, output)
    marker = output / "ready.json.tmp"
    write_json(marker, {"model_manifest_sha256": sha256(run / "manifest.json"),
                        "evaluation_manifest_sha256": sha256(output / "manifest.json")})
    marker.replace(output / "ready.json")
    return {"name": name, "model": str(run), "evaluation": str(output), "summary": summary}


def read_experiment(name):
    """Read a finished run only after checking its model, corpus and evaluation bindings."""
    run, output = experiment_paths(name)
    ready = read_json(output / "ready.json")
    model_signature, eval_signature = sha256(run / "manifest.json"), sha256(output / "manifest.json")
    if ready != {"model_manifest_sha256": model_signature, "evaluation_manifest_sha256": eval_signature}:
        raise ValueError("Experiment completion record does not match its manifests")
    model = read_json(run / "manifest.json")
    evaluation = read_json(output / "manifest.json")
    if model["status"] != "complete" or model["experiment_name"] != name or evaluation["status"] != "complete":
        raise ValueError("Experiment is incomplete")
    if (model["base_model_manifest_sha256"] != sha256(RUN / "manifest.json")
            or model["dataset_manifest_sha256"] != sha256(DATA / "manifest.json")
            or model["corpus_sha256"] != sha256(DATA / "documents.jsonl")):
        raise ValueError("Experiment data or reference model does not match")
    if (evaluation["model_manifest_sha256"] != model_signature
            or evaluation["queries_sha256"] != sha256(ROOT / "resources" / "queries.json")
            or evaluation["test_documents"] != model["test_documents"]
            or evaluation["methods"] != METHODS or evaluation["k"] != 10 or evaluation["source"] != "reuters_topics"):
        raise ValueError("Experiment evaluation does not match the current comparison protocol")
    model_files = {*SHARED_INPUTS, "word_neighbors.csv",
                   *(f"{model_name}/{filename}" for model_name in ARCHITECTURES for filename in WORD_MODEL_FILES)}
    check_hashes(run, model["artifacts_sha256"], model_files)
    check_hashes(ROOT, model["source_sha256"], {*PRODUCER_FILES, EXPERIMENT_SOURCE})
    scoring_files = {path for path in PRODUCER_FILES if path.endswith(".py")} | {"ir_hw4/evaluation.py"}
    check_hashes(ROOT, {path: digest for path, digest in evaluation["source_sha256"].items() if path in scoring_files}, scoring_files)
    check_hashes(output, evaluation["artifacts_sha256"], EVALUATION_FILES)
    return run, output, model, read_json(output / "results.json")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a named CBOW / Skip-Gram experiment")
    subcommands = parser.add_subparsers(dest="command", required=True)
    training = subcommands.add_parser("train", help="Train both models and save a new evaluation")
    training.add_argument("name", help="A unique experiment name")
    for field, (lower, upper, help_text) in PARAMETERS.items():
        default = getattr(DEFAULT_CONFIG, field)
        training.add_argument("--" + field.replace("_", "-"), default=default, type=type(default),
                              help=f"{help_text} ({lower}–{upper}, default {default})")
    args = parser.parse_args()
    try:
        result = train_experiment(args.name, {field: getattr(args, field) for field in PARAMETERS})
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
