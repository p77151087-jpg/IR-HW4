"""Compare equal-weight and TF-IDF-weighted retrieval for saved Word2Vec runs."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .config import ROOT
from .evaluation import metrics, query_bank
from .experiment_ui import REFERENCE_RUNS, checked_bytes, checked_scorecard, scorecard_inputs
from .experiments import EXPERIMENT_SOURCE, experiment_paths
from .io import read_json, sha256, write_json
from .retrieval import ordered_indices, unit_rows, weighted_documents
from .text import document_text, terms
from .training import PRODUCER_FILES, load_engine

COMPARISON = ROOT / "reports" / "embedding-weighting-v2"
QUERY_VARIANTS = ROOT / "resources" / "query_variants.json"
SOURCE_FILES = (*PRODUCER_FILES, EXPERIMENT_SOURCE, "ir_hw4/evaluation.py", "ir_hw4/embedding_weighting.py")
QUERY_STATUS = ("valid_query", "valid_documents", "embedding_weight_coverage")
QUERY_FIELDS = ("p10", "mrr", "ndcg10", *QUERY_STATUS)


def search_query_bank():
    original = query_bank()
    supplement = read_json(QUERY_VARIANTS)
    if supplement["original_queries_sha256"] != sha256(ROOT / "resources" / "queries.json"):
        raise ValueError("補充查詢與原始查詢題庫的版本不符。")
    bank = []
    for query in original:
        qid = query["qid"]
        bank.append({**query, "base_qid": qid, "kind": "keyword"})
        for kind, suffix in (("fuzzy", "F"), ("mixed", "M")):
            variant = supplement["variants"][qid][kind]
            bank.append({**query, **variant, "qid": f"{qid}-{suffix}", "base_qid": qid, "kind": kind})
    return bank


def comparison_inputs():
    model_inputs, queries_sha256 = scorecard_inputs()
    runs = {}
    for name in REFERENCE_RUNS:
        model, evaluation = experiment_paths(name)
        binding = {"model_manifest_sha256": sha256(model / "manifest.json"),
                   "evaluation_manifest_sha256": sha256(evaluation / "manifest.json")}
        if read_json(evaluation / "ready.json") != binding:
            raise ValueError(f"Experiment completion record differs for {name}")
        runs[name] = binding
    return {"model_inputs": model_inputs, "queries_sha256": queries_sha256,
            "query_variants_sha256": sha256(QUERY_VARIANTS), "runs": runs}


def unweighted_scores(engine, method, document_counts, query_counts, query_tokens, relevant_sets):
    words = engine.word_matrices[method]
    document_vectors, _ = weighted_documents(document_counts, words)
    query_vectors, _ = weighted_documents(query_counts, words)
    valid_documents = np.any(document_vectors, axis=1)
    valid_queries = np.any(query_vectors, axis=1)
    scores = np.clip(unit_rows(query_vectors) @ unit_rows(document_vectors).T, -1., 1.)
    vocabulary = set(engine.models[method][0])
    rows = []
    for index, (tokens, relevant) in enumerate(zip(query_tokens, relevant_sets, strict=True)):
        valid_query = bool(valid_queries[index])
        order = ordered_indices(scores[index], engine.ids, valid_documents) if valid_query else ()
        rows.append({**metrics([engine.ids[i] for i in order], relevant),
                     "valid_query": valid_query, "valid_documents": int(valid_documents.sum()),
                     "missing_terms": sorted(set(tokens) - vocabulary)})
    return rows


def weighted_query_score(engine, method, text, relevant):
    ranked, status = engine.rank(text, method)
    return {**metrics([item["id"] for item in ranked], relevant), **{field: status[field] for field in QUERY_STATUS}}


def build_comparison(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite saved comparison: {output}")
    inputs = comparison_inputs()
    sources = {path: sha256(ROOT / path) for path in SOURCE_FILES}
    base, _ = load_engine()
    queries = search_query_bank()
    fixed_qids = [query["qid"] for query in queries if query["kind"] == "keyword"]
    query_tokens = [terms(query["text"]) for query in queries]
    relevant = [{doc["id"] for doc in base.documents if query["topic"] in doc["topics"]} for query in queries]
    if not all(relevant):
        raise ValueError("Each benchmark query must have a known positive test document")

    # Setting every IDF to one preserves token counts and the original vocabulary alignment.
    count_space = replace(base.space)
    count_space.idf.fill(1.)
    document_counts = count_space.transform([terms(document_text(doc)) for doc in base.documents])
    query_counts = count_space.transform(query_tokens)
    tfidf_queries = base.space.transform(query_tokens)
    for query, row in zip(queries, tfidf_queries, strict=True):
        query["tfidf_weights"] = {base.space.vocabulary[index]: float(weight)
                                  for index, weight in zip(row.indices, row.data, strict=True)}

    runs = []
    for name in REFERENCE_RUNS:
        print(f"Evaluating {name}", flush=True)
        run, manifest, weighted = checked_scorecard(name, inputs["model_inputs"], inputs["queries_sha256"])
        engine, _ = load_engine(run)
        if engine.ids != base.ids or engine.space.vocabulary != base.space.vocabulary:
            raise ValueError(f"Test candidates or vocabulary differ for {name}")
        for method in engine.models:
            unweighted = unweighted_scores(engine, method, document_counts, query_counts, query_tokens, relevant)
            saved = {row["qid"]: row for row in weighted["per_query"] if row["method"] == method}
            paired = {}
            for query, before, relevant_ids in zip(queries, unweighted, relevant, strict=True):
                qid = query["qid"]
                if query["kind"] == "keyword":
                    after = saved[qid]
                    if not after["included"]:
                        raise ValueError(f"Fixed query was excluded for {name} {method} {qid}")
                else:
                    after = weighted_query_score(engine, method, query["text"], relevant_ids)
                if after["relevant_total"] != before["relevant_total"]:
                    raise ValueError(f"Relevance sets differ for {name} {method} {qid}")
                paired[qid] = {"unweighted": before, "weighted": {field: after[field] for field in QUERY_FIELDS}}
            summary = next(row for row in weighted["summary"] if row["method"] == method)
            weighted_average = float(np.mean([paired[qid]["weighted"]["ndcg10"] for qid in fixed_qids]))
            if summary["queries_included"] != len(fixed_qids) or not np.isclose(weighted_average, summary["ndcg10"], atol=1e-12, rtol=0):
                raise ValueError(f"Weighted scorecard does not match the fixed queries for {name} {method}")
            runs.append({"run": name, "method": method, "config": manifest["models"][method]["config"],
                         "unweighted_ndcg10": float(np.mean([paired[qid]["unweighted"]["ndcg10"] for qid in fixed_qids])),
                         "weighted_ndcg10": weighted_average, "per_query": paired})

    if inputs != comparison_inputs() or any(sha256(ROOT / path) != digest for path, digest in sources.items()):
        raise ValueError("Comparison inputs changed during evaluation")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "results.json", {"test_documents": len(base.ids), "queries": queries, "runs": runs})
    pending = output / "manifest.json.tmp"
    write_json(pending, {"status": "complete", "inputs": inputs, "source_sha256": sources,
                         "results_sha256": sha256(output / "results.json")})
    pending.replace(output / "manifest.json")


def read_comparison(output=COMPARISON):
    manifest = read_json(output / "manifest.json")
    if manifest["status"] != "complete" or manifest["inputs"] != comparison_inputs():
        raise ValueError("Comparison inputs no longer match the saved result")
    if manifest["source_sha256"] != {path: sha256(ROOT / path) for path in SOURCE_FILES}:
        raise ValueError("Comparison scoring sources changed")
    return json.loads(checked_bytes(output, "results.json", manifest["results_sha256"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=COMPARISON)
    build_comparison(parser.parse_args().output)
