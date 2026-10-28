"""The CIFAR-10 ResNet-9 as a Gradio app: an image (any size, centre-cropped to a square and area-averaged down to
32x32, the network's input), its five likeliest classes. The model is ``CIFARQ_MODEL`` (a hub repo id or a local
directory; default the published model). The examples are the first test image of each class, shown 4x nearest-
neighbour, which the area-averaging takes back to the original pixels exactly."""
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from cifarq.hub import ResNet9

MODEL = os.environ.get('CIFARQ_MODEL', 'thearn/cifar10-resnet9')
net = ResNet9.from_pretrained(MODEL).eval()
EXAMPLES = sorted(str(p) for p in (Path(__file__).parent / 'examples').glob('*.png'))


def to_input(image: Image.Image) -> np.ndarray:
    image = image.convert('RGB')
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    small = image.crop((left, top, left + side, top + side)).resize((32, 32), Image.Resampling.BOX)
    return np.asarray(small)


def classify(image, flip):
    if image is None:
        return None, None
    x = to_input(image)
    probs = net.predict(x[None], tta_flip=flip)[0]
    return {c: float(p) for c, p in zip(net.classes, probs)}, Image.fromarray(x).resize((128, 128), Image.Resampling.NEAREST)


with gr.Blocks(title='CIFAR-10 ResNet-9') as demo:
    gr.Markdown(f'# CIFAR-10 ResNet-9\nA ResNet-9 trained on [CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10) '
                f'with hyperparameters found by Optuna ([model card](https://huggingface.co/{MODEL})). '
                'The network sees 32x32 pixels, so a photo is centre-cropped and shrunk first: the second image is what it sees.')
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='pil', label='Image')
            flip = gr.Checkbox(value=net.tta_flip, label='Average with the mirror image')
            button = gr.Button('Classify', variant='primary')
        with gr.Column():
            label = gr.Label(num_top_classes=5, label='Classes')
            seen = gr.Image(label='Network input (32x32)', interactive=False)
    gr.Examples(EXAMPLES, inputs=image)
    gr.on([button.click, image.upload], classify, [image, flip], [label, seen], api_name='classify')

if __name__ == '__main__':
    demo.launch()
