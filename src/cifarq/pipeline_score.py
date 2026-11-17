"""Complete-pipeline validation and warmed host-to-host timing, without a model-family dependency."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .efficiency import Measurement, score
from .paths import sha256

DATASET_SHA256 = '56cff0f67835b54960f47a6a6bd8d47e27e5d76791c2346206bf48ce1d0509f9'


def load_development(data: str, expected_hash: str = DATASET_SHA256) -> dict:
    """Only materialize training and validation arrays; never read test arrays."""
    if sha256(Path(data)) != expected_hash:
        raise ValueError('dataset hash does not match the frozen evaluation data')
    with np.load(data, allow_pickle=False) as archive:
        return {key: archive[key] for key in ('x_train', 'y_train', 'x_val', 'y_val')}


def classes(output: np.ndarray, count: int) -> np.ndarray:
    output = np.asarray(output)
    if not np.isfinite(output).all():
        raise ValueError('nonfinite predictions')
    if output.shape == (count, 10):
        output = output.argmax(axis=1)
    if output.shape != (count,) or not np.isin(output, np.arange(10)).all():
        raise ValueError('predictor must return ten-class scores or class IDs')
    return output.astype(np.int64)


def evaluate(
    predict: Callable,
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch: int = 256,
    warmup: int = 20,
    repeats: int = 100,
) -> dict:
    """Time the callable itself, including transfers and preprocessing, with real validation batches."""
    if batch < 1 or warmup < 0 or repeats < 1 or len(images) != len(labels) or not len(images):
        raise ValueError('invalid evaluation inputs')
    correct = 0
    for start in range(0, len(images), batch):
        x = images[start : start + batch]
        correct += int((classes(predict(x), len(x)) == labels[start : start + batch]).sum())
    full_batches = [images[start : start + batch] for start in range(0, len(images) - batch + 1, batch)]
    if not full_batches:
        raise ValueError('timing needs at least one full batch')
    for i in range(warmup):
        classes(predict(full_batches[i % len(full_batches)]), batch)
    timings = []
    for i in range(repeats):
        x = full_batches[i % len(full_batches)]
        start = time.perf_counter()
        output = predict(x)
        elapsed = (time.perf_counter() - start) * 1000
        classes(output, len(x))
        timings.append(elapsed)
    return {
        'accuracy': correct / len(labels),
        'latency': statistics.median(timings),
        'latency_ms_samples': timings,
        'batch': batch,
        'warmup': warmup,
        'repeats': repeats,
        'timing_scope': 'decoded host uint8 RGB to host class IDs; preprocessing and transfers included',
    }


def objective(measured: dict, reference: dict) -> float:
    """The adopted primary metric, fixed weights and reference=100."""
    return 100 * score(
        Measurement(measured['accuracy'], measured['latency'], measured['size']),
        Measurement(reference['accuracy'], reference['latency'], reference['size']),
        alpha=0.25,
        beta=0.10,
    )
