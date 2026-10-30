# Data

- **Source**: [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) on the Hugging Face hub, its
  `plain_text/` parquet files at revision `0b2714987fa478483af9968de7c934580d0bb9a2` (`data/manifests/raw.json`
  records the files' sha256; they are the files the first studies were made from).
- **Licence**: CIFAR-10 (Krizhevsky, 2009) is distributed for research without a formal licence; the published model
  states its training data.
- **Rows and ids**: one 32x32 RGB image and its label (10 classes); the id is the index in the hub's file,
  `train:<i>` or `test:<i>`.
- **Cleaning**: decoding only (PNG bytes to uint8 arrays); wrong-sized images or unknown labels are refused, and any
  refusal fails the stage, since CIFAR-10 has none (`data/manifests/clean.json`).
- **Splits**: validation is the first 5000 of `numpy.random.default_rng(1234).permutation` over the 50000 training
  images, training the other 45000, both in that order; test is the hub's 10000 (`data/manifests/dataset.json`).
- **Known issues**: a small share of mislabelled and near-duplicate images, as in every copy of CIFAR-10.
