"""Train a new experiment with independently configured Word2Vec models."""

import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path
import shutil
from time import perf_counter

from gensim.models import Word2Vec
import numpy as np
from scipy import sparse

from .analysis import checked_evaluation
from .config import ANALYSIS_WORDS, DATA, ROOT, RUN
from .evaluation import save_evaluation
from .experiments import ARCHITECTURES, EXPERIMENT_SOURCE, SHARED_INPUTS, check_hashes, experiment_config, experiment_paths
from .io import read_json, sha256, stable_hash, write_csv, write_json
from .retrieval import neighbors, weighted_documents
from .training import EpochRecorder, load_engine


def model_configs(settings):
    if set(settings) != set(ARCHITECTURES):
        raise ValueError("請分別提供 CBOW 與 Skip-Gram 的訓練參數。")
    return {name: replace(experiment_config(settings[name]), sg=sg) for name, sg in ARCHITECTURES.items()}


class ProgressRecorder(EpochRecorder):
    def __init__(self, name, report):
        super().__init__(name)
        self.report = report

    def on_epoch_end(self, model):
        super().on_epoch_end(model)
        self.report("training", model=self.name, epoch=len(self.rows), epochs=model.epochs)


def prepared_sentences(run, base):
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
    return sentences, len(training_ids)


def train_word_model(run, name, options, sentences, train_count, space, lexical, report):
    report("training", model=name, epoch=0, epochs=options.epochs)
    started = perf_counter()
    model = Word2Vec(**asdict(options), hashfxn=stable_hash, hs=0, batch_words=10000)
    model.build_vocab(sentences)
    if len(model.wv) < 2:
        raise ValueError(f"{name} 的 min_count 太高，訓練詞表剩不到兩個詞。")
    initial_digest = hash_bytes(model.wv.vectors.tobytes()).hexdigest()
    callback = ProgressRecorder(name, report)
    effective, raw = model.train(sentences, total_examples=model.corpus_count, epochs=options.epochs,
                                 compute_loss=True, callbacks=[callback])
    words, vectors = model.wv.index_to_key, model.wv.vectors
    if vectors.shape != (len(words), options.vector_size) or not np.isfinite(vectors).all() or len(callback.rows) != options.epochs:
        raise ValueError(f"Incomplete {name} training")
    report("saving", model=name, epoch=options.epochs, epochs=options.epochs)
    target = run / name
    target.mkdir()
    model.save(str(target / "word2vec.model"), separately=[])
    reloaded = Word2Vec.load(str(target / "word2vec.model"))
    if words != reloaded.wv.index_to_key or not np.array_equal(vectors, reloaded.wv.vectors):
        raise ValueError(f"{name} reload mismatch")
    np.save(target / "vectors.npy", vectors, allow_pickle=False)
    write_json(target / "vocabulary.json", words)
    write_csv(target / "epoch_losses.csv", callback.rows)
    averaged, coverage = weighted_documents(lexical, space.word_matrix(words, vectors))
    np.save(target / "test_document_vectors.npy", averaged, allow_pickle=False)
    np.save(target / "test_weight_coverage.npy", coverage, allow_pickle=False)
    metadata = {
        "architecture": "Skip-Gram" if options.sg else "CBOW", "config": asdict(options),
        "vocabulary": len(words), "seconds": perf_counter() - started,
        "initial_vectors_sha256": initial_digest, "offline_reload_exact": True,
        "train_documents": train_count, "test_documents_used_for_training": 0,
        "effective_training_words_all_epochs": int(effective), "raw_training_words_all_epochs": int(raw),
        "initialization": "Fresh Word2Vec trained on the verified train-only sentence sequences.",
    }
    write_json(target / "metadata.json", metadata)
    return metadata, (words, vectors)


def train_parameter_experiment(name, settings, report):
    options = model_configs(settings)
    run, output = experiment_paths(name)
    if run.exists() or output.exists():
        raise FileExistsError(f"實驗 {name} 已存在，請使用新名稱。")
    report("preparing")
    base_engine, base = load_engine(RUN)
    base_signature = sha256(RUN / "manifest.json")
    source = Path(__file__).resolve().relative_to(ROOT.resolve()).as_posix()
    sources = {**base["source_sha256"], EXPERIMENT_SOURCE: sha256(ROOT / EXPERIMENT_SOURCE),
               source: sha256(ROOT / source)}
    checked_evaluation(base_signature)
    if sha256(DATA / "manifest.json") != base["dataset_manifest_sha256"]:
        raise ValueError("The data manifest differs from the prepared training inputs")
    for model, config in options.items():
        if np.count_nonzero(base_engine.space.cf >= config.min_count) < 2:
            raise ValueError(f"{model} 的 min_count 太高，訓練詞表剩不到兩個詞。")

    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "manifest.json", {"status": "training", "experiment_name": name,
                                      "configs": {key: asdict(config) for key, config in options.items()}})
    sentences, train_count = prepared_sentences(run, base)
    lexical = sparse.load_npz(run / "test_tfidf.npz")
    metadata, word_models = {}, {}
    for model, config in options.items():
        metadata[model], word_models[model] = train_word_model(
            run, model, config, sentences, train_count, base_engine.space, lexical, report
        )
    with (run / "word_neighbors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("anchor", "model", "rank", "word", "cosine"))
        writer.writeheader()
        for word in ANALYSIS_WORDS:
            for model, vectors in word_models.items():
                for rank, row in enumerate(neighbors(*vectors, word), 1):
                    writer.writerow({"anchor": word, "model": model, "rank": rank, **row})

    report("evaluating")
    manifest = {
        **base, "status": "complete", "experiment_name": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_manifest_sha256": base_signature,
        "embedding": "Shared TF-IDF weights with independently trained CBOW and Skip-Gram vectors.",
        "models": metadata,
        "source_sha256": sources,
        "artifacts_sha256": {p.relative_to(run).as_posix(): sha256(p) for p in sorted(run.rglob("*"))
                             if p.is_file() and p.name != "manifest.json"},
    }
    write_json(run / "manifest.json", manifest)
    engine, _ = load_engine(run)
    save_evaluation(engine, run, output)
    marker = output / "ready.json.tmp"
    write_json(marker, {"model_manifest_sha256": sha256(run / "manifest.json"),
                        "evaluation_manifest_sha256": sha256(output / "manifest.json")})
    marker.replace(output / "ready.json")
