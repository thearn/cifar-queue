# Torch against JAX profiling (docs/PLAN-jax.md, step 1)

The same configurations (`configs.py`: the champion, Page's recipe, the defaults) on the same card, written to
`results/profile/<framework>-<config>-<mode>.json`. Modes: `e2e` (4 epochs: first and steady epoch, images a second,
peak memory), `components` (augment, forward, forward-backward, train step, evaluation at one batch), `trace` (GPU kernel
time by name from `torch.profiler` / `jax.profiler`).

The Torch side runs the last Torch commit from a worktree, in the Torch venv:

    git worktree add --detach ../cifar-torch f7ab6f5 && ln -s "$PWD/data/dataset" ../cifar-torch/data/dataset
    ML_PROJECT_ROOT=../cifar-torch PYTHONPATH=../cifar-torch/src ~/.venvs/cifarq/bin/python pipeline/tools/profile/torch_prof.py champion e2e
    ~/.venvs/cifarq-jax/bin/python pipeline/tools/profile/jax_prof.py champion e2e
    ~/.venvs/cifarq-jax/bin/python pipeline/tools/profile/jax_prof.py champion components --tag cmdbuf --xla "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN"

Run them with the GPU to themselves (not `runq exclusive` while another session's solves use the card: it pauses
that session's queue).
