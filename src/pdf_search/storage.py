"""Index persistence and search.

FAISS stores only vectors and int64 row ids, so chunk metadata lives in a JSONL
sidecar whose line order *is* the index row order. That invariant is asserted on
load rather than assumed, because a mismatch would return confident, wrong
provenance instead of an error.

The index is `IndexFlatIP` over L2-normalised vectors, so the score returned to
the caller is a cosine similarity in [-1, 1] where higher is better. FAISS's L2
metric returns a *squared* distance where lower is better, which would invert
the ordering implied by the API contract.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from pdf_search.schemas import ChunkRecord, Manifest, SearchResult

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.jsonl"
MANIFEST_FILENAME = "manifest.json"

_FAISS_EMPTY_SLOT = -1


class IndexUnavailableError(RuntimeError):
    """The index is missing, incomplete, or inconsistent with its metadata."""


@dataclass(frozen=True)
class LoadedIndex:
    """An index plus the metadata and manifest that describe it."""

    index: faiss.Index
    chunks: list[ChunkRecord]
    manifest: Manifest

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Return the top_k most similar chunks, most similar first."""
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        wanted = min(top_k, self.index.ntotal)
        if wanted == 0:
            return []

        scores, row_ids = self.index.search(np.ascontiguousarray(query_vector), wanted)

        results: list[SearchResult] = []
        for score, row_id in zip(scores[0], row_ids[0]):
            if row_id == _FAISS_EMPTY_SLOT:
                continue
            chunk = self.chunks[int(row_id)]
            results.append(
                SearchResult(
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=float(np.clip(score, -1.0, 1.0)),
                    text=chunk.text,
                )
            )
        return results


def build_index(vectors: np.ndarray, dim: int) -> faiss.Index:
    """Create a flat inner-product index over unit vectors.

    Flat is exact and, at this corpus size, equivalent to a numpy dot product.
    It is kept for its persistence API and because it is the migration path to
    HNSW without changing this module's interface.
    """
    index = faiss.IndexFlatIP(dim)
    if len(vectors):
        index.add(np.ascontiguousarray(vectors.astype(np.float32)))
    return index


def save(
    output_dir: Path,
    index: faiss.Index,
    chunks: list[ChunkRecord],
    manifest: Manifest,
) -> None:
    """Write the snapshot to a temp directory, validate it, then replace output_dir.

    Ingestion and serving are separate commands, so the API is not running
    during a rebuild. This is not an atomic directory swap -- a portable one
    does not exist -- but a half-written snapshot never becomes the live one.
    """
    if index.ntotal != len(chunks):
        raise ValueError(f"index has {index.ntotal} vectors but {len(chunks)} chunks were given")

    staging = output_dir.with_name(f"{output_dir.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    faiss.write_index(index, str(staging / INDEX_FILENAME))
    with (staging / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
    (staging / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    _validate(staging)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    staging.replace(output_dir)


def _validate(directory: Path) -> None:
    """Reload a freshly written snapshot and confirm it is internally consistent."""
    loaded = load(directory)
    if loaded.index.ntotal != len(loaded.chunks):
        raise ValueError("written snapshot is inconsistent; refusing to publish it")


def load(directory: Path) -> LoadedIndex:
    """Load a snapshot, refusing anything inconsistent.

    Raises IndexUnavailableError so the API can turn it into an actionable 503
    rather than a traceback.
    """
    index_path = directory / INDEX_FILENAME
    metadata_path = directory / METADATA_FILENAME
    manifest_path = directory / MANIFEST_FILENAME

    missing = [p.name for p in (index_path, metadata_path, manifest_path) if not p.exists()]
    if missing:
        raise IndexUnavailableError(
            f"missing {', '.join(missing)} in {directory}. "
            "Run the ingestion command first: python -m pdf_search.ingest "
            "--input-dir <pdf folder> --output-dir <storage folder>"
        )

    index = faiss.read_index(str(index_path))
    with metadata_path.open(encoding="utf-8") as handle:
        chunks = [ChunkRecord.model_validate_json(line) for line in handle if line.strip()]
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    if index.ntotal != len(chunks):
        raise IndexUnavailableError(
            f"index holds {index.ntotal} vectors but metadata holds {len(chunks)} chunks; "
            "the snapshot is inconsistent. Rebuild the index."
        )
    if index.d != manifest.embedding_dim:
        raise IndexUnavailableError(
            f"index dimension {index.d} does not match manifest dimension "
            f"{manifest.embedding_dim}. Rebuild the index."
        )

    return LoadedIndex(index=index, chunks=chunks, manifest=manifest)
