"""The data stages, each a function of files to files that records what it read and wrote (``data/manifests/``):

- ``ingest.fetch``: CIFAR-10's parquet files from the hub at a pinned revision (``configs/data.toml``) to ``data/raw/``;
- ``clean.clean_shard``: one parquet file decoded to one array file (uint8 images, labels);
- ``validate``: checks between the stages, raising on data that would waste a training;
- ``dataset.build``: the train, validation and test arrays in one file; ``dataset.load`` reads it for training.
"""
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
