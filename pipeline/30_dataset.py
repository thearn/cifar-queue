"""Stage 3: the decoded files split into train, validation and test (configs/data.toml, [split]: the fixed
permutation every study so far used), checked (validate.dataset) and written as one array file with
data/manifests/dataset.json. Commit the manifest: it is what every later result names as its data."""
import argparse
import json

from common import record
from cifarq.data import dataset

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    m = dataset.build(force=a.force)
    print(json.dumps({k: m[k] for k in ('file', 'sha256', 'rows')}, indent=1))
    print('saved', record('dataset', {'manifest': m}))
