"""Train-only IDF and three document retrieval methods; no lexical gate for embeddings."""
from collections import Counter
from dataclasses import dataclass
import numpy as np
from scipy import sparse

from .config import METHODS
from .text import terms


def unit_rows(matrix):
    if sparse.issparse(matrix):
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        return (sparse.diags(np.divide(1., norms, out=np.zeros_like(norms), where=norms > 0)) @ matrix).tocsr()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=np.float64), where=norms > 0)


@dataclass
class TfidfSpace:
    vocabulary: list
    df: np.ndarray
    cf: np.ndarray
    n_train: int

    def __post_init__(self):
        self.positions = {word: i for i, word in enumerate(self.vocabulary)}
        self.idf = np.log10(self.n_train / np.asarray(self.df, dtype=np.float64))

    @classmethod
    def fit(cls, sequences):
        df, cf = Counter(), Counter()
        for tokens in sequences:
            counts = Counter(tokens)
            df.update(counts.keys())
            cf.update(counts)
        vocabulary = sorted(df)
        return cls(vocabulary, np.array([df[t] for t in vocabulary]), np.array([cf[t] for t in vocabulary]), len(sequences))

    def transform(self, sequences):
        rows, columns, values = [], [], []
        for row, tokens in enumerate(sequences):
            for word, tf in Counter(tokens).items():
                if word in self.positions:
                    i = self.positions[word]
                    weight = tf * self.idf[i]
                    if weight:
                        rows.append(row)
                        columns.append(i)
                        values.append(weight)
        return sparse.csr_matrix((values, (rows, columns)), shape=(len(sequences), len(self.vocabulary)), dtype=np.float64)

    def word_matrix(self, words, vectors):
        aligned = np.zeros((len(self.vocabulary), vectors.shape[1]), dtype=np.float64)
        for word, vector in zip(words, vectors):
            if word in self.positions:
                aligned[self.positions[word]] = vector
        return aligned


def weighted_documents(tfidf_matrix, aligned_words):
    covered = np.linalg.norm(aligned_words, axis=1) > 0
    mass = np.asarray(tfidf_matrix[:, covered].sum(axis=1)).ravel()
    total = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
    summed = np.asarray(tfidf_matrix @ aligned_words)
    averaged = np.divide(summed, mass[:, None], out=np.zeros_like(summed), where=mass[:, None] > 0)
    coverage = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    return averaged, coverage


def ordered_indices(scores, ids, valid=None):
    valid = np.ones(len(ids), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    # Cosine can be negative. Include every valid document, even scores <= 0.
    available = np.flatnonzero(valid & np.isfinite(scores))
    order = np.lexsort((np.array([int(ids[i]) for i in available]), -np.round(scores[available], 12)))
    return available[order]


class RetrievalEngine:
    def __init__(self, space, test_docs, lexical_matrix, models, document_vectors):
        self.space, self.documents, self.models = space, test_docs, models
        self.ids = [d["id"] for d in test_docs]
        self.by_id = {d["id"]: d for d in test_docs}
        self.lexical = unit_rows(lexical_matrix)
        self.word_matrices = {name: space.word_matrix(*model) for name, model in models.items()}
        self.doc_vectors = {name: unit_rows(values) for name, values in document_vectors.items()}
        self.valid = {"tfidf": np.asarray(self.lexical.multiply(self.lexical).sum(axis=1)).ravel() > 0,
                      **{name: np.linalg.norm(values, axis=1) > 0 for name, values in self.doc_vectors.items()}}

    def score(self, query, method):
        if method not in METHODS:
            raise ValueError("Unknown retrieval method")
        tokens = terms(query)
        matrix = self.space.transform([tokens])
        covered = 1.0
        if method == "tfidf":
            query_vector = unit_rows(matrix)
            valid_query = query_vector.nnz > 0
            scores = (self.lexical @ query_vector.T).toarray().ravel()
        else:
            vector, coverage = weighted_documents(matrix, self.word_matrices[method])
            query_vector = unit_rows(vector)
            valid_query = bool(np.linalg.norm(query_vector) > 0)
            covered = float(coverage[0])
            scores = (self.doc_vectors[method] @ query_vector[0]).ravel()
        known = [word for word in tokens if word in self.space.positions]
        status = {"tokens": tokens, "oov": sorted(set(tokens) - self.space.positions.keys()),
                  "known_tokens": len(known), "total_tokens": len(tokens), "valid_query": valid_query,
                  "embedding_weight_coverage": covered, "valid_documents": int(self.valid[method].sum())}
        # Clip numerical overshoot only. Signed cosine is retained.
        return np.clip(scores, -1., 1.), status

    def rank(self, query, method, limit=None):
        scores, status = self.score(query, method)
        if not status["valid_query"]:
            return [], status
        order = ordered_indices(scores, self.ids, self.valid[method])
        if limit is not None:
            order = order[:limit]
        rows = [{"rank": rank, "id": self.ids[i], "score": float(scores[i]),
                 "title": self.documents[i]["title"] or "（無標題）"} for rank, i in enumerate(order, 1)]
        return rows, status


def neighbors(words, vectors, word, k=5):
    if word not in words:
        return []
    index = {term: i for i, term in enumerate(words)}
    units = unit_rows(np.asarray(vectors, dtype=np.float64))
    anchor = index[word]
    if not np.linalg.norm(units[anchor]):
        return []
    scores = units @ units[anchor]
    candidates = [i for i in range(len(words)) if i != anchor and np.linalg.norm(units[i])]
    candidates.sort(key=lambda i: (-round(float(scores[i]), 12), words[i]))
    return [{"word": words[i], "cosine": float(np.clip(scores[i], -1, 1))} for i in candidates[:k]]
