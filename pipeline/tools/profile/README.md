# JAX profiling

The same configurations (`configs.py`: the champion, Page's recipe, the defaults) on one card, written to
`results/profile/jax-<config>-<mode>.json`. Modes: `e2e` (4 epochs: first and steady epoch, images a second, peak
memory), `components` (augment, forward, forward-backward, train step, evaluation at one batch), `trace` (GPU kernel
time by name from `jax.profiler`).

    ~/.venvs/cifarq-jax/bin/python pipeline/tools/profile/jax_prof.py champion e2e
    ~/.venvs/cifarq-jax/bin/python pipeline/tools/profile/jax_prof.py champion components --tag cmdbuf --xla "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN"

The Torch comparison these tools were written for is answered (docs/PLAN-jax.md, 2026-11-10); the last Torch code is
commit `f7ab6f5`. Run them with the GPU to themselves (not `runq exclusive` while another session's solves use the
card: it pauses that session's queue).
