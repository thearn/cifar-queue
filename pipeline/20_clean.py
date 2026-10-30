"""Stage 2: each raw parquet file decoded to uint8 arrays (configs/data.toml, [clean]), one job per file on the
stage's targets (configs/compute.toml: in this process by default), then checked (validate.cleaned: both splits there,
nothing dropped) and recorded in data/manifests/clean.json. A no-op when the raw files and the rules have not
changed."""
import argparse
import json

from common import executor, record, stage_targets, targets_arg
from cifarq.data import clean, ingest

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    targets_arg(ap)
    a = ap.parse_args()
    targets = stage_targets('clean', a.targets)
    with executor('clean', targets, data=ingest.shards()) as ex:
        m = clean.run(ex, force=a.force)
    print(json.dumps({k: m[k] for k in ('rows', 'kept', 'dropped')}, indent=1))
    print('saved', record('clean', {'targets': targets, 'manifest': m}))
