"""Stage 7: publish to the Hugging Face hub (configs/publish.toml): the model repo holds this repository at its
committed HEAD, the champion's weights at its root and README.md as the model card (pipeline/card.md: the dataset's
hash, the seeds, the commit); the Space holds app/. Each upload mirrors its folder (files gone here are deleted
there), its commit message the local commit. Refuses a tree with uncommitted changes: the model repo is HEAD.

    python pipeline/70_publish.py --dry-run          # stage the model repo in results/publish/ and stop
    python pipeline/70_publish.py --private          # first time: both repos private, made public after a look
    python pipeline/70_publish.py --no-space         # the model repo only (a Gradio Space needs PRO or ZeroGPU)
"""
import argparse
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from string import Template

from common import RESULTS, ROOT, latest, record
from cifarq import paths

WEIGHTS = ('model.safetensors', 'config.json', 'hyperparameters.json')


def card(readme: str, champion: dict, publish: dict, dataset: dict, raw: dict, commit: str, space: bool) -> str:
    """The model card: card.md's metadata and model section around the repository README's introduction and the rest
    of it."""
    c = champion['chosen']
    pct = lambda v, d=1: f'{v * 100:.{d}f} %'
    runs = '\n'.join(f"| {r['seed']} | {pct(r['val_acc'])} | {pct(r['test_acc'])} | {r['device']} |" for r in champion['runs'])
    demo = f" [The demo](https://huggingface.co/spaces/{publish['space_repo']}) classifies uploaded images." if space else ''
    text = Template((Path(__file__).parent / 'card.md').read_text()).substitute(
        license=publish['license'], pipeline_tag=publish['pipeline_tag'], tags=', '.join(publish['tags']),
        dataset=publish['dataset'], name=publish['model_repo'].split('/')[-1], revision=raw['source'].get('revision'),
        test_acc=c['test_acc'], source=champion['source'], n_seeds=len(champion['runs']), seed=c['seed'],
        val_pct=pct(c['val_acc']), test_pct=pct(c['test_acc']), test_mean=f"{champion['test_mean'] * 100:.2f}",
        test_stdev=pct(champion['test_stdev'], 2), runs=runs, demo=demo, dataset_file=dataset['file'],
        dataset_sha=dataset['sha256'], commit=commit)
    intro, _, rest = readme.partition('\n## ')                # the README after its introduction
    return text.replace('{{README_INTRO}}', intro.rstrip() + '\n').replace('{{README_REST}}', '## ' + rest.rstrip() + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='stage the model repo in results/publish/, upload nothing')
    ap.add_argument('--no-space', action='store_true', help='publish the model repo only')
    ap.add_argument('--private', action='store_true', help='create the repos private (only when they do not exist)')
    ap.add_argument('--champion', default=None, help='a results/champion/*.json (default: the latest)')
    a = ap.parse_args()
    git = lambda *args: subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, check=True).stdout
    if git('status', '--porcelain', '--untracked-files=no').strip() and not a.dry_run:
        raise SystemExit('uncommitted changes: commit first, the model repo is HEAD')
    commit = git('rev-parse', '--short', 'HEAD').decode().strip()
    champion = json.loads(Path(a.champion).read_text()) if a.champion else latest('champion')
    publish, dataset = paths.config('publish'), paths.read_manifest('dataset')
    trained_on = champion.get('provenance', {}).get('dataset')
    if trained_on != dataset['sha256']:
        raise SystemExit(f"the champion was trained on dataset {trained_on}, the manifest names "
                         f"{dataset['sha256']}: train it again (60_champion.py), or check out the manifest it used")
    stage = RESULTS / 'publish' / 'model'
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    tarfile.open(fileobj=io.BytesIO(git('archive', '--format=tar', 'HEAD'))).extractall(stage, filter='data')
    for f in WEIGHTS:
        shutil.copy2(Path(champion['chosen']['model']) / f, stage / f)
    (stage / 'README.md').write_text(card((stage / 'README.md').read_text(), champion, publish, dataset,
                                          paths.read_manifest('raw'), commit, space=not a.no_space))
    if a.dry_run:
        print(f'staged in {stage}')
        raise SystemExit(0)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(publish['model_repo'], private=a.private, exist_ok=True)
    out = {'model': str(api.upload_folder(repo_id=publish['model_repo'], folder_path=stage, delete_patterns='*',
                                          commit_message=f'{ROOT.name} {commit}: champion of {champion["time"]}'))}
    if not a.no_space:
        api.create_repo(publish['space_repo'], repo_type='space', space_sdk='gradio', private=a.private, exist_ok=True)
        out['space'] = str(api.upload_folder(repo_id=publish['space_repo'], repo_type='space', folder_path=ROOT / 'app',
                                             delete_patterns='*', ignore_patterns=['__pycache__/*', '*.pyc'],
                                             commit_message=f'{ROOT.name} {commit}'))
    print(json.dumps(out, indent=1))
    print('saved', record('publish', {'commit': commit, 'champion': champion['time'], **out}))
