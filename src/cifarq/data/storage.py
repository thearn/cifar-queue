"""Reading a source and writing to a bucket, by URI: a local path, ``gs://bucket/prefix``, ``https://...`` (one file),
or ``hf://datasets/OWNER/NAME@REVISION/path`` (a file or folder of a hub dataset at a pinned revision). ``fetch``
returns the files it wrote and what it resolved (the hub's revision when none was given), for the manifest."""
from __future__ import annotations

import fnmatch
import shutil
import urllib.request
from pathlib import Path


def fetch(uri: str, dest: Path, pattern: str = '*') -> tuple[list[Path], dict]:
    """Copy the files at ``uri`` matching ``pattern`` into ``dest``; (the files, {"uri", "revision"?})."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if uri.startswith('gs://'):
        return _fetch_gcs(uri, dest, pattern), {'uri': uri}
    if uri.startswith('hf://'):
        return _fetch_hf(uri, dest, pattern)
    if uri.startswith(('http://', 'https://')):
        out = dest / uri.rstrip('/').rsplit('/', 1)[-1]
        with urllib.request.urlopen(uri) as r, open(out, 'wb') as f:
            shutil.copyfileobj(r, f)
        return [out], {'uri': uri}
    src = Path(uri).expanduser()
    files = [src] if src.is_file() else sorted(p for p in src.rglob('*') if p.is_file() and fnmatch.fnmatch(p.name, pattern))
    out = []
    for p in files:
        target = dest / (p.name if src.is_file() else p.relative_to(src))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        out.append(target)
    return out, {'uri': str(src.resolve())}


def put(files: list[Path], uri: str, base: Path) -> list[str]:
    """Upload ``files`` (paths under ``base``) to ``uri`` (``gs://bucket/prefix`` or a local directory); their URIs."""
    out = []
    if uri.startswith('gs://'):
        bucket, prefix = _gcs_bucket(uri)
        for f in files:
            name = f'{prefix}{Path(f).relative_to(base).as_posix()}'
            bucket.blob(name).upload_from_filename(str(f))
            out.append(f'gs://{bucket.name}/{name}')
        return out
    root = Path(uri).expanduser()
    for f in files:
        target = root / Path(f).relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        out.append(str(target))
    return out


def _gcs_bucket(uri: str):
    from google.cloud import storage                      # pip install -e '.[gcs]'
    name, _, prefix = uri[len('gs://'):].partition('/')
    return storage.Client().bucket(name), (prefix.rstrip('/') + '/' if prefix else '')


def _fetch_gcs(uri: str, dest: Path, pattern: str) -> list[Path]:
    bucket, prefix = _gcs_bucket(uri)
    out = []
    for blob in bucket.client.list_blobs(bucket, prefix=prefix):
        rel = blob.name[len(prefix):]
        if rel and not rel.endswith('/') and fnmatch.fnmatch(rel.rsplit('/', 1)[-1], pattern):
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))
            out.append(target)
    return out


def _fetch_hf(uri: str, dest: Path, pattern: str) -> tuple[list[Path], dict]:
    from huggingface_hub import HfApi, snapshot_download
    kind, _, rest = uri[len('hf://'):].partition('/')         # datasets/OWNER/NAME@REV/path
    owner, name, *path = rest.split('/')
    name, _, revision = name.partition('@')
    repo = f'{owner}/{name}'
    repo_type = {'datasets': 'dataset', 'models': 'model', 'spaces': 'space'}[kind]
    revision = revision or HfApi().repo_info(repo, repo_type=repo_type).sha
    sub = '/'.join(path)
    local = Path(snapshot_download(repo, repo_type=repo_type, revision=revision,
                                   allow_patterns=[f'{sub}/**/{pattern}', f'{sub}/{pattern}', sub] if sub else [pattern, f'**/{pattern}']))
    files, _ = fetch(str(local / sub) if sub else str(local), dest, pattern)
    return files, {'uri': f'hf://{kind}/{repo}@{revision}/{sub}', 'revision': revision}
