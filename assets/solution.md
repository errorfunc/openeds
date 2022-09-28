## Solution description

In general solution is based on two-stage approach (Estimator → Forecaster). End-to-end approach did not work out on the leaderboard.

### Preprocessing

The original dataset comes in 640 x 400 resolution. We've managed to make things work without quality loss over 256 x 160 images. It improved overall speed of training and inference which was essential under our hardware constraints. 

### Estimator 
#### (Algorithm that estimates gaze vector given image)

Based on the amount of data and some test runs, vanilla ResNet-18 was chosen (we tinkered only input channels because dataset images were grayscale). Known modifications of ResNet blocks (ResNet-B, ResNet-C, Resnet-D) did not improve our results.

At training no fancy stuff was used. AdamW optimizer combined with conservative LR scheduling and slightly tuned early stopping did the best in the terms of loss. 
Unfortunately we've faced serious overfitting, so hard augmentation strategy was involved into our training pipeline. SWA did improve score a bit, but was rejected due to the complication of the code base (original source code was built on the pytorch-lightning framework which did not have SWA callback at that time). 


In the end, we've got excellent and robust estimator (with angular error such as _0.00019 rad_). 
### Augmentations
For this purpose we chose to modify **[albumentations](https://albumentations.ai/)** library, so affine transformations were applicable to the gaze vector itself. 

![](aug_example.png "Examples of same image gone through augmentations" )



### Forecaster 
#### (Algorithm that predicts upcoming gaze vectors based on the history of previous frames)

This is the tricky part. An orthodox strategy is to train some kind of recurrent network. 
Apparently an approach suffers from multiple flaws such as instability, need for large amount of data and so on. 

Instead, we chose to incorporate knowledge of different kinds of eye movement. 
In challenge's time constraints the only reasonable types of movements are saccades and fixations, 
smooth pursuit takes too long to complete (~100 ms). 
So main idea is straightforward. 
All we need is to find if there exists saccade movement in any phase just before the moment of prediction, 
otherwise just use the latest gaze vectors "as is" because an eye is still in fixation phase.

With such a logic simple heuristic may be developed: check if fast change in gradient exists and if so, 
just add computed gradient to predicted frames. 
Otherwise, compute moving average of last frames and use the result as prediction.

We tried to use different methods for forecasting (exponential smoothing, linear regression, Savitzky-Golay filter) but all of them failed to beat our approach.


### Hardware
Best solution was trained on quite old machine (i7-7700k + single 1080ti), so distributed training is not implemented. 
As a part of another project there exists much smaller version of this estimator,
but it's not presented here because it wasn’t used in the competition.


[Back to README](../README.md)