"""Explicit offline training. Importing this module never trains or downloads."""
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
from scipy import sparse
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from .config import ROOT, RUN, ANALYSIS_WORDS
from .dataset import load_data
from .io import read_json, write_json, write_csv, sha256, stable_hash
from .text import terms, document_text, document_blocks
from .sentences import split_sentences
from .retrieval import TfidfSpace, weighted_documents, RetrievalEngine, neighbors

PRODUCER_FILES = ["ir_hw4/config.py", "ir_hw4/dataset.py", "ir_hw4/io.py", "ir_hw4/text.py",
                  "ir_hw4/sentences.py", "ir_hw4/retrieval.py", "ir_hw4/training.py", "resources/stopwords.json"]


@dataclass(frozen=True)
class TrainingConfig:
    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    epochs: int = 30
    seed: int = 42
    workers: int = 1
    sg: int = 0
    negative: int = 5
    sample: float = 0.001
    ns_exponent: float = 0.75
    alpha: float = 0.025
    min_alpha: float = 0.0001
    cbow_mean: int = 1
    sorted_vocab: int = 1
    shrink_windows: bool = True


class EpochRecorder(CallbackAny2Vec):
    def __init__(self, name):
        self.name, self.rows, self.previous = name, [], 0.

    def on_epoch_begin(self, model):
        self.started = perf_counter()

    def on_epoch_end(self, model):
        loss = float(model.get_latest_training_loss())
        self.rows.append({"epoch": len(self.rows) + 1, "loss_delta": loss - self.previous,
                          "loss_cumulative": loss, "seconds": perf_counter() - self.started})
        self.previous = loss
        if len(self.rows) % 5 == 0:
            print(f"{self.name}: epoch {len(self.rows)}/{model.epochs}", flush=True)


def train(run=RUN):
    run = Path(run).resolve()
    if not run.is_relative_to(ROOT.resolve()):
        raise ValueError("Training outputs must stay in HW4")
    if run.exists():
        raise FileExistsError("Refuse to overwrite models; use a new --run directory")
    documents, data_manifest = load_data()
    training = [d for d in documents if d["split"] == "train"]
    testing = [d for d in documents if d["split"] == "test"]
    train_tokens = [terms(document_text(d)) for d in training]
    test_tokens = [terms(document_text(d)) for d in testing]
    run.mkdir(parents=True)
    write_json(run / "manifest.json", {"status": "training"})
    sentences, sentence_ids = [], []
    with (run / "training_sentences.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for doc, expected in zip(training, train_tokens):
            flattened = []
            for block in document_blocks(doc):
                for sentence in split_sentences(block):
                    tokens = terms(sentence.text)
                    if tokens:
                        sentences.append(tokens)
                        sentence_ids.append(doc["id"])
                        flattened.extend(tokens)
                        stream.write(json.dumps({"id": doc["id"], "tokens": tokens}, ensure_ascii=False) + "\n")
            if flattened != expected:
                raise ValueError(f"Sentence splitting changed tokens for document {doc['id']}")
    write_json(run / "document_order.json", {"train": [d["id"] for d in training], "test": [d["id"] for d in testing]})
    space = TfidfSpace.fit(train_tokens)
    lexical = space.transform(test_tokens)
    sparse.save_npz(run / "test_tfidf.npz", lexical)
    np.savez(run / "term_statistics.npz", df=space.df, cf=space.cf)
    write_json(run / "vocabulary.json", space.vocabulary)
    write_csv(run / "term_statistics.csv",
              ({"term": word, "train_tf": int(space.cf[i]), "train_df": int(space.df[i]), "train_idf": float(space.idf[i])}
               for i, word in enumerate(space.vocabulary)))
    metadata, models, doc_vectors = {}, {}, {}
    for name, sg in (("cbow", 0), ("skipgram", 1)):
        config = replace(TrainingConfig(), sg=sg)
        print(f"Training new {name}: {len(sentences)} sentences; train-only", flush=True)
        started = perf_counter()
        model = Word2Vec(**asdict(config), hashfxn=stable_hash, hs=0, batch_words=10000)
        model.build_vocab(sentences)
        initial_hash = __import__("hashlib").sha256(model.wv.vectors.tobytes()).hexdigest()
        callback = EpochRecorder(name)
        effective, raw = model.train(sentences, total_examples=model.corpus_count, epochs=config.epochs,
                                     compute_loss=True, callbacks=[callback])
        if not np.isfinite(model.wv.vectors).all() or len(callback.rows) != config.epochs:
            raise ValueError("Incomplete training")
        target = run / name
        target.mkdir()
        model.save(str(target / "word2vec.model"), separately=[])
        vectors = np.asarray(model.wv.vectors)
        words = model.wv.index_to_key
        np.save(target / "vectors.npy", vectors, allow_pickle=False)
        write_json(target / "vocabulary.json", words)
        write_csv(target / "epoch_losses.csv", callback.rows)
        loaded = Word2Vec.load(str(target / "word2vec.model"))
        if loaded.wv.index_to_key != words or not np.array_equal(loaded.wv.vectors, vectors):
            raise ValueError("Model reload mismatch")
        averaged, coverage = weighted_documents(lexical, space.word_matrix(words, vectors))
        np.save(target / "test_document_vectors.npy", averaged, allow_pickle=False)
        np.save(target / "test_weight_coverage.npy", coverage, allow_pickle=False)
        metadata[name] = {"architecture": "Skip-Gram" if sg else "CBOW", "config": asdict(config),
                          "vocabulary": len(words), "seconds": perf_counter() - started,
                          "initial_vectors_sha256": initial_hash, "offline_reload_exact": True,
                          "train_documents": len(training), "test_documents_used_for_training": 0,
                          "effective_training_words_all_epochs": int(effective), "raw_training_words_all_epochs": int(raw),
                          "initialization": "New Gensim Word2Vec constructor and build_vocab; no pretrained/resumed weights.",
                          "loss_note": "Differences of cumulative negative-sampling loss, not semantic accuracy."}
        write_json(target / "metadata.json", metadata[name])
        models[name], doc_vectors[name] = (words, vectors), averaged
    neighbors_rows = []
    for word in ANALYSIS_WORDS:
        for name, (words, vectors) in models.items():
            for rank, neighbor in enumerate(neighbors(words, vectors, word), 1):
                neighbors_rows.append({"anchor": word, "model": name, "rank": rank, **neighbor})
    write_csv(run / "word_neighbors.csv", neighbors_rows)
    manifest = {
        "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_sha256": data_manifest["documents_sha256"],
        "dataset_manifest_sha256": sha256(ROOT / "data/reuters-21578/manifest.json"),
        "training_sentences": len(sentences), "training_tokens": sum(map(len, sentences)),
        "train_documents": len(training), "test_documents": len(testing),
        "test_tokens": sum(map(len, test_tokens)), "train_vocabulary": len(space.vocabulary),
        "empty_train_documents": sum(not t for t in train_tokens), "empty_test_documents": sum(not t for t in test_tokens),
        "fitting": "Train-only vocabulary, DF, IDF and Word2Vec. No unused/test text or topics used for training.",
        "text_policy": "TITLE plus BODY, no topics/date/IDs in scoring. Shared C-style preprocessing, fixed stopwords, no stemming.",
        "tfidf": "tf * log10(N_train / DF_train); actual repeated query TF; test/query OOV ignored.",
        "embedding": "Sum of TF-IDF * original Word2Vec vector / sum of weights with embedding; document/query use the same rule; cosine in 100 dimensions.",
        "ranking": "All valid test documents; retain zero/negative cosine; zero-vector documents excluded and counted; score rounded to 12 decimals only for stable ordering, tie by numeric NEWID; saved scores full precision.",
        "models": metadata, "source_sha256": {path: sha256(ROOT / path) for path in PRODUCER_FILES},
        "versions": {**{package: version(package) for package in ("numpy", "scipy", "gensim", "regex", "streamlit", "scikit-learn")},
                     "python": platform.python_version()},
        "artifacts_sha256": {p.relative_to(run).as_posix(): sha256(p) for p in sorted(run.rglob("*")) if p.is_file() and p.name != "manifest.json"}}
    write_json(run / "manifest.json", manifest)
    return manifest


def load_engine(run=RUN, verify=True):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Training is not complete")
    if verify:
        for relative, expected in manifest["artifacts_sha256"].items():
            if sha256(run / relative) != expected:
                raise ValueError(f"Model artifact SHA256 mismatch: {relative}")
        for relative, expected in manifest["source_sha256"].items():
            if sha256(ROOT / relative) != expected:
                raise ValueError(f"Model-producing source has changed: {relative}. Rebuild in a new --run.")
        for package, expected in manifest["versions"].items():
            if package != "python" and version(package) != expected:
                raise ValueError(f"Dependency version mismatch: {package}")
    documents, data_manifest = load_data()
    if data_manifest["documents_sha256"] != manifest["corpus_sha256"]:
        raise ValueError("Model and corpus versions differ")
    order = read_json(run / "document_order.json")
    by_id = {d["id"]: d for d in documents}
    test_docs = [by_id[doc_id] for doc_id in order["test"]]
    if any(d["split"] != "test" for d in test_docs) or set(order["train"]) & set(order["test"]):
        raise ValueError("Invalid saved train/test split")
    statistics = np.load(run / "term_statistics.npz", allow_pickle=False)
    space = TfidfSpace(read_json(run / "vocabulary.json"), statistics["df"], statistics["cf"], manifest["train_documents"])
    models, vectors = {}, {}
    for name in ("cbow", "skipgram"):
        models[name] = (read_json(run / name / "vocabulary.json"), np.load(run / name / "vectors.npy", allow_pickle=False))
        vectors[name] = np.load(run / name / "test_document_vectors.npy", allow_pickle=False)
    return RetrievalEngine(space, test_docs, sparse.load_npz(run / "test_tfidf.npz"), models, vectors), manifest
