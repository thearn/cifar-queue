"""The Gradio app classifies an image with a saved network (skipped without gradio)."""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from cifarq import train
from conftest import SMALL


def test_app_classifies(built, tmp_path, monkeypatch):
    pytest.importorskip('gradio')
    monkeypatch.chdir(tmp_path)
    r = train.train(epochs=1, save='model', **SMALL)
    monkeypatch.setenv('MODEL', str(tmp_path / r['saved']))
    spec = importlib.util.spec_from_file_location('app', Path(__file__).resolve().parents[1] / 'app' / 'app.py')
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    probs, seen = app.classify(Image.fromarray(built['x_test'][0]).resize((96, 96)), False)
    assert set(probs) == set(app.net.classes) and seen.size == (128, 128)
