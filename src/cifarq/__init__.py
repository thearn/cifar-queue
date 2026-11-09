"""CIFAR-10 with a small CNN, trained through the machine's solve queue (../../runq): one training, Optuna on a VM,
and Optuna here distributing trials over the local GPU and Colab VMs."""

import os

# JAX takes most of a GPU's memory at its first use unless told not to; trials share cards (runq's device turns, two
# trainings at once on a G4) and the local card with other jobs, so it allocates as it goes.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

__version__ = '0.3.0'
