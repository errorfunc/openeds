## 1st place solution for OpenEDS Gaze Prediction Challenge

## Table of contents
- [Challenge description](assets/challenge.md)
- [Solution description](assets/solution.md)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Links](#links)

#### Installation
`pip install git+https://github.com/errorfunc/openeds.git`


#### Usage
These are main commands of this package:

* **openeds-resize** -- resizes original images to desired resolution 
* **openeds-assemble** -- assembles dataset suitable for learning/testing
* **openeds-train** -- trains ResNet-18 model on the dataset
* **openeds-infer** -- infers ResNet-18 model on the dataset
* **openeds-forecast** -- predicts five frames into future 50 ms
* **openeds-ensemble** -- makes simple average of selected forecasts

#### Example
Imagine that we want to train ResNet-18 model:
1. `openeds-resize -in='openeds_dataset/train' -out='data/train'`
2. `openeds-assemble -in='data/train' -out='data/train.csv'`
3. `openeds-train -in='data/train.csv' -out='models'`


or to predict gaze vectors on the test dataset:

1. `openeds-assemble -in='data/test' -out='data/test.csv'`
2. `openeds-infer -in='data/test.csv' -out='data/test_estimation.csv' -c='resnet.pt'`
3. `openeds-forecast -in='data/test_estimation.csv' -out='resnet.json'`

or to create ensemble of multiple forecasts:

`openeds-ensemble -in='data/submits/' -out='data/best_submission.json'`

#### Links

* Best models are available [here](https://bit.ly/3tWA4Kl)
* Dataset's [description](https://arxiv.org/abs/2005.03876)
* Dataset itself is available upon request [here](http://research.fb.com/programs/openeds-2020-challenge/)
* [Paper](https://www.mdpi.com/1424-8220/21/14/4769) describing results of the competition
* Our [presentation](https://youtu.be/lgvx_RwhH6Q)