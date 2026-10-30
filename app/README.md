---
title: CIFAR-10 ResNet-9
emoji: 🖼️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: apache-2.0
models:
  - thearn/cifar10-resnet9
datasets:
  - uoft-cs/cifar10
short_description: A ResNet-9 tuned by Optuna on CIFAR-10
---

A demo of [thearn/cifar10-resnet9](https://huggingface.co/thearn/cifar10-resnet9), a ResNet-9 trained on
[CIFAR-10](https://huggingface.co/datasets/uoft-cs/cifar10). Upload any image, or pick one of the test-set examples;
since the network was trained on 32x32 images, photos are centre-cropped and shrunk to that size first, which costs
a good deal of accuracy on anything that does not look like a CIFAR-10 image.
