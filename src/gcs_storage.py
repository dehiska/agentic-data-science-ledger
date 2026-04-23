"""
GCS Storage helper — upload/download CSV datasets for the Swarm AutoML pipeline.

In cloud mode (APP_MODE=cloud), CSVs are stored in a GCS bucket so the
backend can access them from Cloud Run.  In local mode the bucket is unused
and local paths are passed directly.

Environment variables:
  GCS_BUCKET   — name of the GCS bucket (e.g. "agentic-ledger-datasets-myproject")
                 Set automatically via Cloud Run env var.

Usage:
    from src.gcs_storage import upload_csv, download_to_tmp, is_gcs_path
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

GCS_BUCKET = os.getenv("GCS_BUCKET", "")
_GCS_PREFIX = "swarm-datasets"


def is_gcs_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("gs://")


def upload_csv(local_path: str, dest_name: Optional[str] = None) -> str:
    """
    Upload a local CSV file to GCS.

    Returns the gs:// URI of the uploaded file.
    Raises RuntimeError if GCS_BUCKET is not set.
    """
    if not GCS_BUCKET:
        raise RuntimeError(
            "GCS_BUCKET environment variable is not set. "
            "Set it in your Cloud Run env vars or .env file."
        )

    dest_name = dest_name or Path(local_path).name
    blob_path = f"{_GCS_PREFIX}/{dest_name}"

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

    gcs_uri = f"gs://{GCS_BUCKET}/{blob_path}"
    return gcs_uri


def upload_bytes(data: bytes, dest_name: str) -> str:
    """
    Upload raw bytes (e.g. from st.file_uploader) directly to GCS.

    Returns the gs:// URI.
    """
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET environment variable is not set.")

    blob_path = f"{_GCS_PREFIX}/{dest_name}"

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type="text/csv")

    return f"gs://{GCS_BUCKET}/{blob_path}"


def download_to_tmp(gcs_uri: str) -> str:
    """
    Download a gs:// URI to a local temporary file.

    Returns the local file path.  The caller is responsible for
    deleting the file when done (use finally: Path(path).unlink()).
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {gcs_uri}")

    # Parse gs://bucket/path/to/file.csv
    without_prefix = gcs_uri[5:]
    bucket_name, blob_path = without_prefix.split("/", 1)
    filename = Path(blob_path).name

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_path)

    tmp_file = tempfile.NamedTemporaryFile(
        suffix=f"_{filename}", delete=False
    )
    tmp_file.close()
    blob.download_to_filename(tmp_file.name)
    return tmp_file.name


def list_datasets() -> list:
    """Return list of CSV filenames currently in the GCS bucket."""
    if not GCS_BUCKET:
        return []
    try:
        from google.cloud import storage
        client = storage.Client()
        blobs  = client.list_blobs(GCS_BUCKET, prefix=f"{_GCS_PREFIX}/")
        return [
            {"name": Path(b.name).name, "gcs_uri": f"gs://{GCS_BUCKET}/{b.name}",
             "size_mb": round(b.size / 1e6, 2)}
            for b in blobs
            if b.name.endswith(".csv")
        ]
    except Exception:
        return []
