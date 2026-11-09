"""Independent scalar/NumPy numerical audit of the sealed formal experiment."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec

from .config import ROOT, RUN, DATA, METHODS
from .dataset import load_data, ReutersParser
from .evaluation import query_bank, evaluate
from .io import read_json, write_json, sha256
from .text import terms, document_text
from .training import load_engine


def assert_close(left, right, tolerance=1e-10):
    difference = float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    if not np.isfinite(difference) or difference > tolerance:
        raise AssertionError(f"Numerical mismatch: {difference}")
    return difference


def verify(run=RUN):
    engine, manifest = load_engine(run, verify=True)
    documents, data_manifest = load_data()
    raw_documents = []
    for name, expected in data_manifest["raw_sha256"].items():
        assert sha256(DATA / name) == expected
        parser = ReutersParser()
        parser.feed((DATA / name).read_bytes().decode("latin-1"))
        raw_documents.extend(parser.documents)
    assert sorted(raw_documents, key=lambda d: int(d["id"])) == documents
    assert sha256(DATA / "reuters21578.tar.gz") == data_manifest["archive_sha256"]
    assert sha256(DATA / "README.original.txt") == data_manifest["readme_sha256"]
    train_docs = [d for d in documents if d["split"] == "train"]
    train_counts = {d["id"]: Counter(terms(document_text(d))) for d in train_docs}
    test_counts = [Counter(terms(document_text(d))) for d in engine.documents]
    df, cf = Counter(), Counter()
    for counts in train_counts.values():
        df.update(counts.keys())
        cf.update(counts)
    assert sorted(df) == engine.space.vocabulary
    assert list(engine.space.df) == [df[word] for word in engine.space.vocabulary]
    assert list(engine.space.cf) == [cf[word] for word in engine.space.vocabulary]
    idf = {word: math.log10(len(train_docs) / frequency) for word, frequency in df.items()}
    sentence_counts = defaultdict(Counter)
    for line in (Path(run) / "training_sentences.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert row["id"] in train_counts
        sentence_counts[row["id"]].update(row["tokens"])
    assert all(sentence_counts[key] == counts for key, counts in train_counts.items())
    weights = [{word: tf * idf[word] for word, tf in counts.items() if word in idf} for counts in test_counts]
    norms = np.array([math.sqrt(sum(w*w for w in vector.values())) for vector in weights])
    lexical_error, embedding_error, vector_error, metrics_error = 0., 0., 0., 0.
    references = {}
    for method in ("cbow", "skipgram"):
        words, values = engine.models[method]
        lookup = dict(zip(words, values.astype(np.float64)))
        model = Word2Vec.load(str(Path(run) / method / "word2vec.model"))
        assert model.wv.index_to_key == words and np.array_equal(model.wv.vectors, values)
        def average(vector):
            covered = [(word, weight) for word, weight in vector.items() if word in lookup and weight > 0]
            if not covered:
                return np.zeros(values.shape[1], dtype=np.float64)
            return np.average(np.stack([lookup[word] for word, _ in covered]),
                              weights=np.array([weight for _, weight in covered]), axis=0)
        averaged = np.array([average(vector) for vector in weights])
        stored = np.load(Path(run) / method / "test_document_vectors.npy", allow_pickle=False)
        vector_error = max(vector_error, assert_close(averaged, stored))
        references[method] = (averaged, lookup)
    comparisons = 0
    for query in query_bank():
        q_counts = Counter(terms(query["text"]))
        q_weights = {word: tf * idf[word] for word, tf in q_counts.items() if word in idf}
        q_norm = math.sqrt(sum(w*w for w in q_weights.values()))
        reference = np.array([sum(weight * q_weights.get(word, 0.) for word, weight in vector.items()) /
                              (norm * q_norm) if norm and q_norm else 0. for vector, norm in zip(weights, norms)])
        scores, _ = engine.score(query["text"], "tfidf")
        lexical_error = max(lexical_error, assert_close(reference, scores))
        for method, (averaged, lookup) in references.items():
            covered = [(word, weight) for word, weight in q_weights.items() if word in lookup and weight > 0]
            q = np.average(np.stack([lookup[word] for word, _ in covered]),
                           weights=np.array([weight for _, weight in covered]), axis=0) if covered else np.zeros(averaged.shape[1])
            denominator = np.linalg.norm(averaged, axis=1) * np.linalg.norm(q)
            reference = np.divide(np.einsum("ij,j->i", averaged, q), denominator,
                                  out=np.zeros(len(averaged)), where=denominator > 0)
            scores, _ = engine.score(query["text"], method)
            embedding_error = max(embedding_error, assert_close(reference, scores))
        for method in METHODS:
            ranking, _ = engine.rank(query["text"], method)
            relevant = {d["id"] for d in engine.documents if query["topic"] in d["topics"]}
            positions = [i for i, row in enumerate(ranking, 1) if row["id"] in relevant]
            expected = [len([i for i in positions if i <= 10]) / 10,
                        1 / positions[0] if positions else 0,
                        sum(1 / math.log2(i + 1) for i in positions if i <= 10) /
                        sum(1 / math.log2(i + 1) for i in range(1, min(10, len(relevant)) + 1))]
            computed = __import__("ir_hw4.evaluation", fromlist=["metrics"]).metrics([r["id"] for r in ranking], relevant)
            metrics_error = max(metrics_error, assert_close(expected, [computed["p10"], computed["mrr"], computed["ndcg10"]]))
            comparisons += len(engine.ids)
    first = evaluate(engine)
    second = evaluate(engine)
    assert first == second, "Repeated experiment results differ"
    return {"status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_sha256": manifest["corpus_sha256"], "model_manifest_sha256": sha256(Path(run) / "manifest.json"),
            "original_sgml_reparsed_exact": True, "train_only_df_cf_and_sentence_counts_exact": True,
            "full_models_reload_exact": True, "artifacts_verified": len(manifest["artifacts_sha256"]),
            "score_comparisons": comparisons, "tfidf_max_error": lexical_error, "embedding_max_error": embedding_error,
            "weighted_document_vector_max_error": vector_error, "metrics_max_error": metrics_error,
            "repeated_evaluation_exact": True, "queries_sha256": sha256(ROOT / "resources/queries.json"),
            "validation_source_sha256": sha256(Path(__file__))}


def verify_hw3():
    before = read_json(ROOT / "reports/hw3-protection-before.json")
    source = Path(before["root"])
    changed = [relative for relative, expected in before["files"].items()
               if not (source / relative).is_file() or sha256(source / relative) != expected]
    if changed:
        raise AssertionError(f"HW3 protected files changed: {changed}")
    return {"status": "unchanged", "protected_files": len(before["files"]), "changed": changed,
            "excluded": before["excluded"], "checked_at_utc": datetime.now(timezone.utc).isoformat()}
