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

import hashlib
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
_DIGEST_BLOCK_BYTES = 1 << 20


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
    """Write the snapshot to a staging area, validate it, then publish it.

    Staging lives *inside* output_dir rather than beside it. The output
    directory is typically a mounted volume whose parent is the container root,
    which a non-root user cannot write to -- a sibling staging directory fails
    with EACCES there.

    Ingestion and serving are separate commands, so the API is not running
    during a rebuild. This is not an atomic directory swap -- a portable one
    does not exist -- but the snapshot is fully written and reloaded before any
    live file is touched, so a failed build never replaces a good index.
    """
    if index.ntotal != len(chunks):
        raise ValueError(f"index has {index.ntotal} vectors but {len(chunks)} chunks were given")

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    faiss.write_index(index, str(staging / INDEX_FILENAME))
    with (staging / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
    sealed = manifest.model_copy(
        update={
            "index_sha256": _digest(staging / INDEX_FILENAME),
            "metadata_sha256": _digest(staging / METADATA_FILENAME),
        }
    )
    (staging / MANIFEST_FILENAME).write_text(sealed.model_dump_json(indent=2), encoding="utf-8")

    _validate(staging)

    for filename in (INDEX_FILENAME, METADATA_FILENAME, MANIFEST_FILENAME):
        (staging / filename).replace(output_dir / filename)
    shutil.rmtree(staging)


def _digest(path: Path) -> str:
    """sha256 of a file, read in blocks so a large index is never held in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_DIGEST_BLOCK_BYTES), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _verify_binding(
    directory: Path, index: faiss.Index, chunks: list[ChunkRecord], manifest: Manifest
) -> None:
    """Confirm these three files are the ones that were written together.

    Row count and dimension agreeing proves only that the shapes match; two
    unrelated snapshots of the same corpus size agree on both. The digests bind
    the files to each other, and the row-position check enforces the sidecar
    ordering this module's contract depends on.
    """
    if manifest.index_sha256 is None or manifest.metadata_sha256 is None:
        raise IndexUnavailableError(
            f"the manifest in {directory} carries no file digest, so the snapshot cannot be "
            "verified. Rebuild the index."
        )

    expected = {
        INDEX_FILENAME: manifest.index_sha256,
        METADATA_FILENAME: manifest.metadata_sha256,
    }
    for filename, digest in expected.items():
        if _digest(directory / filename) != digest:
            raise IndexUnavailableError(
                f"{filename} does not match the digest recorded in the manifest; the snapshot "
                "has been modified or mixed with another. Rebuild the index."
            )

    for row, chunk in enumerate(chunks):
        if chunk.chunk_index != row:
            raise IndexUnavailableError(
                f"metadata line {row} holds chunk_index {chunk.chunk_index}; sidecar order must "
                "match index row order. Rebuild the index."
            )

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise IndexUnavailableError(
            "the index was not built with the inner-product metric, so its scores are not the "
            "cosine similarities the API contract promises. Rebuild the index."
        )


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

    _verify_binding(directory, index, chunks, manifest)

    return LoadedIndex(index=index, chunks=chunks, manifest=manifest)
