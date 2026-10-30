"""Stage 1: CIFAR-10's parquet files from the Hugging Face hub at the revision pinned in configs/data.toml, into
data/raw/, and data/manifests/raw.json (the source, its revision, each file's sha256). A no-op when the source and
settings have not changed (--force: fetch again)."""
import argparse
import json

from common import record
from cifarq import paths
from cifarq.data import ingest, storage

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--push', action='store_true', help='mirror data/raw/ to the configured bucket')
    a = ap.parse_args()
    m = ingest.fetch(force=a.force)
    print(json.dumps({k: m[k] for k in ('source', 'fetched')}, indent=1), f'{len(m["files"])} files')
    if a.push:
        bucket = paths.config('data')['storage']['bucket']
        if not bucket:
            raise SystemExit('no [storage] bucket in configs/data.toml')
        print(len(storage.put(ingest.shards(), f'{bucket.rstrip("/")}/raw/', ingest.RAW)), 'files pushed to', bucket)
    print('saved', record('ingest', {'manifest': m}))
