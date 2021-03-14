# pcnaDeep: a deep-learning based single-cell cycle profiler with PCNA signal

<img src="/bin/assets/icon.png" alt="banner" width="200" align="top" />

Welcome! pcnaDeep integrates cutting-edge detection techniques with tracking and cell cycle resolving models.
With pre-trained Detectron2 maskRCNN model, pcnaDeep is able to detect and resolve very dense PCNA-fluorescent expressing cell tracks.

## Installation
1. PyTorch (torch >= 1.7.1) installation is essential, while CUDA GPU support is recommended. Visit [PyTorch homepage](https://pytorch.org/) for specific installation schedule.
2. Install [Detectron2](https://github.com/facebookresearch/detectron2) (>=0.3)
3. `pip install pcnaDeep`, or from source `python setup.py install`.
4. (optional, for annotation only) Download [VGG Image Annotator 2](https://www.robots.ox.ac.uk/~vgg/software/via/) software.
5. (optional, for evaluation only) Install deepcell-label for annotating tracking ground truth.
6. (optional, for visualisation only) Install [Fiji (ImageJ)](https://fiji.sc/) with [TrackMate CSV Importer](https://github.com/tinevez/TrackMate-CSVImporter) plugin.

## Getting started

