"""Checks between the stages, raising ``DataError`` on data that would waste a training or a search: run after
decoding (how much was dropped, and why) and on the built dataset (shapes, labels, every class in every split, no
image in two splits)."""
from __future__ import annotations

import numpy as np


class DataError(ValueError):
    pass


def cleaned(reports: list[dict], rules: dict) -> None:
    """The decoded files' reports: both splits there, enough rows kept, and no more dropped than ``max_drop_fraction``."""
    splits = {r['split'] for r in reports}
    if splits != {'train', 'test'}:
        raise DataError(f'decoded splits {sorted(splits)}: expected train and test')
    rows = sum(r['rows'] for r in reports)
    kept = sum(r['kept'] for r in reports)
    if kept < rules['min_rows']:
        raise DataError(f'{kept} rows kept, fewer than min_rows = {rules["min_rows"]}')
    if rows and 1 - kept / rows > rules['max_drop_fraction']:
        why = {k: sum(r['dropped'][k] for r in reports) for k in reports[0]['dropped']}
        raise DataError(f'decoding dropped {rows - kept} of {rows} rows (max_drop_fraction = '
                        f'{rules["max_drop_fraction"]}): {why}')


def dataset(d: dict) -> None:
    """A built dataset (``dataset.load``'s arrays)."""
    n_classes = len(d['classes'])
    ids = []
    for split in ('train', 'val', 'test'):
        x, y = d[f'x_{split}'], d[f'y_{split}']
        if len(x) == 0 or len(x) != len(y) or x.shape[1:] != (32, 32, 3) or x.dtype != np.uint8:
            raise DataError(f'{split}: images {x.shape} {x.dtype} against labels {y.shape}')
        missing = set(range(n_classes)) - set(np.unique(y).tolist())
        if missing or y.max() >= n_classes:
            raise DataError(f'{split}: no images of class {sorted(missing)}, or a label past {n_classes - 1}')
        ids.append(d[f'ids_{split}'])
    if len(np.unique(np.concatenate(ids))) != sum(map(len, ids)):
        raise DataError('an image is in more than one split')
