"""The feature space the preference model learns in.

    song -> [ PCA of mean-centered audio embedding | tag scores | signal features ]
         -> each column standardized (mean 0, std 1)

Why each step:
  * mean-centering: all embeddings share a big common direction (Phase 0), which
    would otherwise dominate;
  * PCA: 512 numbers -> ~32, so the preference model needs far less feedback;
  * standardizing: puts BPM (~100) and tag scores (~0.1) on the same scale, so the
    learned weights are comparable ("how much does each feature matter?").

The fitted numbers are saved, so songs added later are transformed identically.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_STD = 1e-8  # constant columns stay 0 instead of dividing by zero


@dataclass(frozen=True)
class FeatureSpace:
    version: str
    embedding_model: str
    embedding_mean: np.ndarray  # (dim,)
    components: np.ndarray  # (k, dim) principal directions
    explained_variance_ratio: np.ndarray  # (k,)
    column_mean: np.ndarray  # (k + n_tabular,)
    column_std: np.ndarray  # (k + n_tabular,)
    names: list[str]  # one per output column

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        tabular: np.ndarray,
        tabular_names: list[str],
        *,
        n_components: int,
        version: str,
        embedding_model: str,
    ) -> "FeatureSpace":
        n, dim = embeddings.shape
        if tabular.shape != (n, len(tabular_names)):
            raise ValueError(
                f"tabular shape {tabular.shape} doesn't match "
                f"{n} songs x {len(tabular_names)} names"
            )
        if not 1 <= n_components <= min(n, dim):
            raise ValueError(f"n_components must be in [1, {min(n, dim)}], got {n_components}")
        if not (np.isfinite(embeddings).all() and np.isfinite(tabular).all()):
            raise ValueError("features contain NaN or inf")

        mean = embeddings.mean(axis=0)
        _, singular_values, vt = np.linalg.svd(embeddings - mean, full_matrices=False)
        components = _fix_signs(vt[:n_components])
        variance = singular_values**2
        pcs = (embeddings - mean) @ components.T
        columns = np.hstack([pcs, tabular])
        return cls(
            version=version,
            embedding_model=embedding_model,
            embedding_mean=mean,
            components=components,
            explained_variance_ratio=variance[:n_components] / variance.sum(),
            column_mean=columns.mean(axis=0),
            column_std=np.maximum(columns.std(axis=0), MIN_STD),
            names=[f"pc{i + 1}" for i in range(n_components)] + list(tabular_names),
        )

    def transform(self, embeddings: np.ndarray, tabular: np.ndarray) -> np.ndarray:
        if embeddings.shape[1] != self.embedding_mean.shape[0]:
            raise ValueError("embedding dimension doesn't match the fitted space")
        if tabular.shape[1] != len(self.names) - len(self.components):
            raise ValueError("number of tabular features doesn't match the fitted space")
        pcs = (embeddings - self.embedding_mean) @ self.components.T
        return (np.hstack([pcs, tabular]) - self.column_mean) / self.column_std

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            version=self.version,
            embedding_model=self.embedding_model,
            embedding_mean=self.embedding_mean,
            components=self.components,
            explained_variance_ratio=self.explained_variance_ratio,
            column_mean=self.column_mean,
            column_std=self.column_std,
            names=np.array(self.names),
        )

    @classmethod
    def load(cls, path: Path) -> "FeatureSpace":
        d = np.load(path)
        return cls(
            version=str(d["version"]),
            embedding_model=str(d["embedding_model"]),
            embedding_mean=d["embedding_mean"],
            components=d["components"],
            explained_variance_ratio=d["explained_variance_ratio"],
            column_mean=d["column_mean"],
            column_std=d["column_std"],
            names=[str(n) for n in d["names"]],
        )


def _fix_signs(components: np.ndarray) -> np.ndarray:
    """SVD may flip a component's sign arbitrarily; make the largest entry positive
    so refitting on the same data always gives the same features."""
    signs = np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])
    return components * signs[:, None]
