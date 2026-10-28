"""Publish to the Hugging Face hub: the model repo (this repository at its committed HEAD, the champion's weights and
config at the root, and README.md as the model card, its metadata naming the dataset and the test accuracy) and the
Space (space/: the Gradio app, which installs cifarq from the model repo). Each upload mirrors its folder (files gone
here are deleted there), its commit message the local commit.

    python scripts/publish.py --private          # first time: both repos private, made public by hand after a look
    python scripts/publish.py                    # later: update both (visibility unchanged)
"""
import argparse
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from common import REPO, RESULTS

MODEL_REPO, SPACE_REPO = 'thearn/cifar10-resnet9', 'thearn/cifar10-resnet9-demo'
WEIGHTS = ('model.safetensors', 'config.json', 'hyperparameters.json')


def card(readme: str, champion: dict, revision: str) -> str:
    c = champion['chosen']
    runs = '\n'.join(f"| {r['seed']} | {r['val_acc'] * 100:.1f} % | {r['test_acc'] * 100:.1f} % | {r['device']} |"
                     for r in champion['runs'])
    meta = f"""---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-classification
tags: [image-classification, cifar10, resnet9, optuna, pytorch_model_hub_mixin, model_hub_mixin]
datasets: [uoft-cs/cifar10]
metrics: [accuracy]
model-index:
- name: cifar10-resnet9
  results:
  - task: {{type: image-classification, name: Image Classification}}
    dataset: {{type: uoft-cs/cifar10, name: CIFAR-10, split: test, revision: {revision}}}
    metrics:
    - {{type: accuracy, value: {c['test_acc']}, name: Test accuracy}}
---
"""
    use = f"""
## The published model

The weights here are the champion of the search ({champion['source']}) retrained over {len(champion['runs'])} seeds,
keeping the seed with the median validation accuracy (seed {c['seed']}: {c['val_acc'] * 100:.1f} % validation,
{c['test_acc'] * 100:.1f} % test). Over the seeds the test accuracy was {champion['test_mean'] * 100:.2f} +-
{champion['test_stdev'] * 100:.2f} %.

| Seed | Validation | Test | GPU |
|---|---|---|---|
{runs}

```python
from cifarq.hub import ResNet9            # pip install git+https://huggingface.co/{MODEL_REPO}
net = ResNet9.from_pretrained('{MODEL_REPO}')
probs = net.predict(images)              # uint8 (n, 32, 32, 3) RGB -> (n, 10); net.classes names the columns
```

`hyperparameters.json` is the full training configuration, and [the demo](https://huggingface.co/spaces/{SPACE_REPO})
classifies uploaded images.
"""
    head, _, rest = readme.partition('\n## ')                # after the introduction
    return meta + head + use + '\n## ' + rest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--champion', default=None, help='a results/champion/*.json (default: the latest)')
    ap.add_argument('--private', action='store_true', help='create the repos private (only when they do not exist)')
    a = ap.parse_args()
    git = lambda *args: subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, check=True).stdout
    if git('status', '--porcelain', '--untracked-files=no').strip():
        raise SystemExit('uncommitted changes: commit first, the model repo is HEAD')
    sha = git('rev-parse', '--short', 'HEAD').decode().strip()
    champion_file = Path(a.champion) if a.champion else sorted((RESULTS / 'champion').glob('*.json'))[-1]
    champion = json.loads(champion_file.read_text())
    model_dir = Path(champion['chosen']['model'])
    revision = json.loads((REPO / 'data' / 'MANIFEST.json').read_text())['revision']

    api = HfApi()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / 'model'
        tarfile.open(fileobj=io.BytesIO(git('archive', '--format=tar', 'HEAD'))).extractall(stage, filter='data')
        for f in WEIGHTS:
            (stage / f).write_bytes((model_dir / f).read_bytes())
        (stage / 'README.md').write_text(card((stage / 'README.md').read_text(), champion, revision))
        api.create_repo(MODEL_REPO, private=a.private, exist_ok=True)
        print(api.upload_folder(repo_id=MODEL_REPO, folder_path=stage, delete_patterns='*',
                                commit_message=f'cifar-queue {sha}: champion {champion_file.stem}'))
    api.create_repo(SPACE_REPO, repo_type='space', space_sdk='gradio', private=a.private, exist_ok=True)
    print(api.upload_folder(repo_id=SPACE_REPO, repo_type='space', folder_path=REPO / 'space', delete_patterns='*',
                            ignore_patterns=['__pycache__/*', '*.pyc'], commit_message=f'cifar-queue {sha}'))
