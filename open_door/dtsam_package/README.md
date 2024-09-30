Detic + SAM for objects recognition and segmentation
=======================================

## Description
It's the DTSAM (Detic + SAM) python package for objects recognition and segmentation.

Specifically, it uses Detic for objects recognition to get the bounding boxes of objects and then uses SAM for object segmentation to get the masks of objects.

The main script is: [detic_sam.py](detic_sam.py).

Details of Detic and SAM can be found in their respective repositories:

- Detic: https://github.com/facebookresearch/detic
- SAM: https://github.com/facebookresearch/detectron2/blob/main/projects/SegmentAnything

## Installation
First add the following to your bash profile (assuming you have CUDA 11+):
`export CUDA_PATH=/usr/local/cuda-11.7/`

Next, be sure to use `python 3.8`. If you have a higher version of python, then install 3.8 and use this.

Either run `./setup.sh` (make sure the `python` command uses python 3.8 in this case!) or follow the steps manually.

Details can be checked in `setup.sh` or `setup_alta.sh`

## Usage
Run `run_detic_sam.sh`
There are four parameters:
- `IMAGE_PATH`: The image path to run the detection on.
- `CLASSES`: The object classes that you want to detect.
- `DEVICE`: The device to run the detection on (e.g. 'cuda:0' or 'cpu').
- `THRESHOLD`: Detection threshold.

More details can be checked in [dtsam.py](../dtsam.py)

## Troubleshooting
- Segmentation Fault (core dumped)
  
  This stems from an issue with the detectron2 installation. Torch and detectron2 are closely linked: you need to make
  sure you've installed the torch version with the right CUDA extension corresponding to the detectron2 version (as well
  as your own system setup). Find the correct detectron2 installation command [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
  Then, find the corresponding torch version compatible withthat and make sure you have that.
- `PIL.Image.LINEAR` doesn't exist.
  ```
    File "/home/nkumar/detic-sam/venv/lib/python3.10/site-packages/detectron2/data/transforms/transform.py", line 46, in ExtentTransform
    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
    AttributeError: module 'PIL.Image' has no attribute 'LINEAR'. Did you mean: 'BILINEAR'?
  ```
  Then simply edit the offending file to change `Image.LINEAR` to `Image.BILINEAR`.

## Acknowledgements
Thank [bdaiinstitute](https://github.com/bdaiinstitute) for providing the initial [code](https://github.com/bdaiinstitute/detic-sam) !