# pcnaDeep: a deep-learning based single-cell cycle profiler with PCNA signal

<img src="/assets/icon.png" alt="banner" width="200" align="top" />

Welcome! pcnaDeep integrates cutting-edge detection techniques with tracking and cell cycle resolving models.
With pre-trained Detectron2 maskRCNN model, pcnaDeep is able to detect and resolve very dense cell tracks with __PCNA fluorescent Only__.

![image](/assets/res_demo.gif)

## Installation
1. PyTorch (torch >= 1.7.1) installation is essential, while CUDA GPU support is recommended. Visit [PyTorch homepage](https://pytorch.org/) for specific installation schedule.
2. Install [Detectron2](https://github.com/facebookresearch/detectron2) (>=0.3)
   - To build Detectron2 on __Windows__ may require the following change of `torch` package. [Reference (Chinese)](https://blog.csdn.net/weixin_42644340/article/details/109178660).
    ```angular2html
       In torch\include\torch\csrc\jit\argument_spec.h,
       static constexpr size_t DEPTH_LIMIT = 128;
          change to -->
       static const size_t DEPTH_LIMIT = 128;
    ```
3. `pip install pcnaDeep`, or from source `python setup.py install`.
4. (optional, for annotation only) Download [VGG Image Annotator 2](https://www.robots.ox.ac.uk/~vgg/software/via/) software.
5. (optional, for visualisation only) Install [Fiji (ImageJ)](https://fiji.sc/) with [TrackMate CSV Importer](https://github.com/tinevez/TrackMate-CSVImporter) plugin.

## Getting started

