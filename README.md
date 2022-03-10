# aug_accuracy

This package contains a set of tools to train and test classification neural networks.

The structure is inspired by msd_pytorch package.

Data loader includes standard augmentation techniques implemented without transforms from torchvision.

Model combines architectures from torchvision with a scaling layer for normalization.

Tensorboard is used for logging.

# Installation
```
conda create -n class_nn -c conda-forge 'pytorch=*=*cuda*' 'torchvision=*=*cu*' tensorboard 'imageio<2.15' scikit-learn scikit-image matplotlib numpy
```
```
pip install -e .
```
