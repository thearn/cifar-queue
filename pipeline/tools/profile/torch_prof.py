"""Torch side: python torch_prof.py CONFIG [e2e|components|trace]. Run with ~/.venvs/cifarq/bin/python and
PYTHONPATH=<torch worktree>/src. Writes results/profile/torch-<config>-<mode>.json."""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from configs import CONFIGS, DATA, kernel_table, write  # noqa: E402

N_WARM, N_TIME = 10, 50


def setup(h):
    import torch

    from cifarq import train
    from cifarq.data import dataset

    h = {**train.DEFAULTS, **h}
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = torch.device('cuda')
    a = dataset.load(DATA)
    gpu = lambda x: torch.from_numpy(x).to(dev)
    xt, yt = gpu(a['x_train']).permute(0, 3, 1, 2).contiguous(), gpu(a['y_train']).long()
    xv, yv = gpu(a['x_val']).permute(0, 3, 1, 2).contiguous(), gpu(a['y_val']).long()
    mean, std = torch.tensor(train.MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(train.STD, device=dev).view(1, 3, 1, 1)
    norm = lambda x: ((x - mean) / std).contiguous(memory_format=torch.channels_last)
    wh = train.whitening_filters(norm(xt[:5000].float() / 255), h['whiten_eps']) if h['whiten'] else None
    net = (
        train.model(h['width'], h['dropout'], h['activation'], celu_alpha=h['celu_alpha'], head=h['head'], logit_scale=h['logit_scale'], whitening=wh)
        .to(dev)
        .to(memory_format=torch.channels_last)
    )
    tr = [p for p in net.parameters() if p.requires_grad]
    groups = [
        {'params': [p for p in tr if p.ndim > 1], 'weight_decay': h['weight_decay']},
        {'params': [p for p in tr if p.ndim <= 1], 'weight_decay': 0.0},
    ]
    if h['optimizer'].endswith('_sf'):
        import schedulefree

        opt = (
            schedulefree.AdamWScheduleFree(groups, lr=h['lr'], betas=(h['beta1'], h['beta2']), warmup_steps=100)
            if h['optimizer'] == 'adamw_sf'
            else schedulefree.SGDScheduleFree(groups, lr=h['lr'], momentum=h['momentum'] or 0.9, warmup_steps=100)
        )
        opt.train()
    elif h['optimizer'] == 'sgd':
        opt = torch.optim.SGD(groups, lr=h['lr'], momentum=h['momentum'], nesterov=bool(h['nesterov']) and h['momentum'] > 0)
    else:
        opt = torch.optim.AdamW(groups, lr=h['lr'], betas=(h['beta1'], h['beta2']))
    return h, net, opt, xt, yt, xv, yv, norm


def timed(fn, sync):
    import torch

    for _ in range(N_WARM):
        fn()
    sync()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(N_TIME):
        fn()
    e.record()
    sync()
    return round(s.elapsed_time(e) / N_TIME, 3)  # ms per call


def components(cfg):
    import torch
    import torch.nn.functional as F

    from cifarq import evaluate, train

    h, net, opt, xt, yt, xv, yv, norm = setup(CONFIGS[cfg])
    bs = h['batch_size']
    idx = torch.randperm(len(yt), device=xt.device)[:bs]
    xb, yb = xt[idx], yt[idx]
    sync = torch.cuda.synchronize

    def aug():
        return train.augment(xb.float() / 255, h)

    def fwd():
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            return net(norm(xb.float() / 255))

    def fwdbwd():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(net(norm(xb.float() / 255)), yb, label_smoothing=h['label_smoothing'])
        loss.backward()

    def step():
        x = train.augment(xb.float() / 255, h)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(net(norm(x)), yb, label_smoothing=h['label_smoothing'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    rec = {
        'framework': 'torch',
        'config': cfg,
        'batch': bs,
        'device': torch.cuda.get_device_name(),
        'ms': {'augment': timed(aug, sync), 'forward': timed(fwd, sync), 'forward_backward': timed(fwdbwd, sync), 'train_step': timed(step, sync)},
    }
    t = time.time()
    for _ in range(3):
        evaluate.accuracy(net, xv, yv, norm, h['tta_flip'])
    sync()
    rec['ms']['eval_val_5000'] = round((time.time() - t) / 3 * 1e3, 1)
    rec['steps_per_epoch'] = len(yt) // bs
    rec['epoch_estimate_s'] = round(rec['ms']['train_step'] * rec['steps_per_epoch'] / 1e3, 2)
    rec['peak_gpu_mb'] = round(torch.cuda.max_memory_allocated() / 2**20)
    write(f'torch-{cfg}-components', rec)


def e2e(cfg):
    import torch

    from cifarq import train

    t = time.time()
    r = train.train(data=DATA, **{**CONFIGS[cfg], 'epochs': 4})
    hist = [e['seconds'] for e in r['history']]
    write(
        f'torch-{cfg}-e2e',
        {
            'framework': 'torch',
            'config': cfg,
            'first_epoch_s': hist[0],
            'steady_epoch_s': round((hist[-1] - hist[0]) / 3, 3),
            'process_wall_s': round(time.time() - t, 1),
            'val_acc_4ep': r['val_acc'],
            'images_per_s': r['images_per_s'],
            'peak_gpu_mb': round(torch.cuda.max_memory_allocated() / 2**20),
        },
    )


def trace(cfg):
    import torch
    import torch.nn.functional as F
    from torch.profiler import ProfilerActivity, profile

    from cifarq import train

    h, net, opt, xt, yt, xv, yv, norm = setup(CONFIGS[cfg])
    bs = h['batch_size']

    def step(i):
        idx = torch.randint(0, len(yt), (bs,), device=xt.device)
        x = train.augment(xt[idx].float() / 255, h)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(net(norm(x)), yt[idx], label_smoothing=h['label_smoothing'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for i in range(10):
        step(i)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for i in range(20):
            step(i)
        torch.cuda.synchronize()
    f = Path(tempfile.mkdtemp()) / 'trace.json'
    p.export_chrome_trace(str(f))
    events = json.loads(f.read_text())['traceEvents']
    write(f'torch-{cfg}-trace', {'framework': 'torch', 'config': cfg, 'steps': 20, 'kernels': kernel_table(events)})


if __name__ == '__main__':
    cfg, mode = sys.argv[1], sys.argv[2]
    {'e2e': e2e, 'components': components, 'trace': trace}[mode](cfg)
