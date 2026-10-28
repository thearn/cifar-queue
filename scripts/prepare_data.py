"""Download CIFAR-10 from the Hugging Face hub and write data/cifar10-<sha>.npz and data/MANIFEST.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cifarq import data  # noqa: E402

if __name__ == '__main__':
    out = data.prepare(sys.argv[1] if len(sys.argv) > 1 else None)
    print(out, out.stat().st_size / 1e6, 'MB')
    print(data.MANIFEST.read_text())
