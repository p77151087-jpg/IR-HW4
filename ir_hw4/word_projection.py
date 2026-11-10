"""Compute and retain full vocabulary projections for an existing model."""

import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from .config import ROOT
from .io import sha256
from .retrieval import unit_rows


PROJECTION_METHODS = ("PCA", "t-SNE")
PROJECTION_CACHE = ROOT / "tmp" / "word-projections"
PROJECTOR_SIGNATURE = f"{sha256(__file__)}:{np.__version__}:{sklearn.__version__}"


def project_neighbor_space(vectors, method):
    """Project the center word and its first five neighbors."""
    units = unit_rows(np.asarray(vectors, dtype=np.float64))
    if method == "t-SNE":
        perplexity = max(1., (len(units) - 1) / 3)
        if len(np.unique(units.round(12), axis=0)) < 3:
            return {"xy": None, "perplexity": perplexity}
        xy = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=42,
                  method="exact").fit_transform(units)
        return {"xy": centered_tsne(xy), "perplexity": perplexity}
    centered = units - units.mean(axis=0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2 / max(1, len(units) - 1)
    total = variance.sum()
    xy = np.zeros((len(units), 2))
    components = min(2, len(singular))
    xy[:, :components] = u[:, :components] * singular[:components]
    leading = np.pad(variance[:components], (0, 2 - components))
    return {"xy": xy, "variance": leading, "variance_ratio": leading / total if total else leading}


def centered_tsne(xy):
    if not np.isfinite(xy).all():
        return None
    return xy - (xy.min(axis=0) + xy.max(axis=0)) / 2


def compute_vocabulary_projection(vectors, method):
    """Return coordinates in vector row order, leaving zero vectors unprojected."""
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0
    units = vectors[valid] / norms[valid, None]
    with threadpool_limits(limits=4, user_api="blas"):
        if len(units) <= 6:
            projection = project_neighbor_space(units, method)
        elif method == "PCA":
            pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
            projection = {"xy": pca.fit_transform(units), "variance": pca.explained_variance_,
                          "variance_ratio": np.nan_to_num(pca.explained_variance_ratio_)}
        else:
            if units.shape[1] > 50:
                pca = PCA(n_components=min(50, len(units) - 1), svd_solver="randomized", random_state=42)
                units = pca.fit_transform(units)
            perplexity = min(30., (len(units) - 1) / 3)
            with threadpool_limits(limits=4, user_api="openmp"):
                xy = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42,
                          method="barnes_hut", max_iter=500, angle=.7, n_jobs=4).fit_transform(units)
            projection = {"xy": centered_tsne(xy), "perplexity": perplexity}
    if projection["xy"] is not None:
        coordinates = np.full((len(vectors), 2), np.nan, dtype=projection["xy"].dtype)
        coordinates[valid] = projection["xy"]
        projection["xy"] = coordinates
    return projection


def projection_cache_path(model_signature, architecture, method):
    if method not in PROJECTION_METHODS:
        raise ValueError(f"Unknown projection: {method}")
    key = f"{PROJECTOR_SIGNATURE}:{model_signature}:{architecture}:{method}"
    return PROJECTION_CACHE / (hashlib.sha256(key.encode()).hexdigest() + ".npz")


def read_cached_projection(path, vectors, method):
    try:
        with np.load(path, allow_pickle=False) as payload:
            if payload["key"].item() != path.stem:
                return None
            xy = payload["xy"]
            if xy.shape != (len(vectors), 2) or not np.issubdtype(xy.dtype, np.floating):
                return None
            valid = np.linalg.norm(vectors, axis=1) > 0
            if not np.isfinite(xy[valid]).all() or not np.isnan(xy[~valid]).all():
                return None
            if method == "PCA":
                variance, ratios = payload["variance"], payload["variance_ratio"]
                if any(values.shape != (2,) or not np.isfinite(values).all() or (values < 0).any()
                       for values in (variance, ratios)) or ratios.sum() > 1.00001:
                    return None
                return {"xy": xy, "variance": variance, "variance_ratio": ratios}
            perplexity = float(payload["perplexity"].item())
            if not 0 < perplexity < int(valid.sum()):
                return None
            return {"xy": xy, "perplexity": perplexity}
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        return None


def write_cached_projection(path, projection):
    """Publish complete arrays atomically so another session can read them safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, key=path.stem, **projection)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_vocabulary_projection(model_signature, architecture, method, vectors):
    path = projection_cache_path(model_signature, architecture, method)
    cached = read_cached_projection(path, vectors, method)
    if cached is not None:
        return cached
    projection = compute_vocabulary_projection(vectors, method)
    if projection["xy"] is not None:
        try:
            write_cached_projection(path, projection)
        except OSError:
            # The live chart can still use the computed result without a writable cache.
            pass
    return projection
