# Plt2Pix: Line Art Colorization Using Palette

This repository is based on [Tag2Pix](http://arxiv.org/abs/1908.05840).
Check the original implementation in [here](https://github.com/blandocs/tag2pix).

## Prerequisite
 * pytorch >= 1.8.1
 * torchvision >= 0.9.0
 * numpy
 * scipy 
 * python-opencv
 * scikit-image
 * Pillow (PIL)
 * imageio
 * tqdm

## Test

Download all network dumps from [releases](https://github.com/jhoowy/plt2pix/releases) and place it in this project root directory.

 * `python test.py --load=plt2pix_50_epoch.pkl`


