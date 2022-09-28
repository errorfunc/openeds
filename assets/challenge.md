### The OpenEDS 2020 Challenge

The OpenEDS 2020 Challenge (https://research.fb.com/programs/openeds-2020-challenge/) was hosted as part of the OpenEyes: Eye gaze
in AR, VR and in the Wild (https://openeyes-workshop.github.io/) workshop, organized at the European Conference on Computer Vision in 2020. 

![](gaze_example.png "Examples of images without glasses (top row) and with glasses (bottom row), representing the variability of the
dataset in terms of accessories, ethnicity, age and gender" )

### Final Leaderboard
| Team Name        | PE   |
|------------------|------|
| random_b         | 3.17 |
| Eyemazing        | 3.31 |
| DmitryAKonovalov | 3.34 |
| baseline | 5.28 |


### Gaze Prediction Challenge description

The task consisted in designing a model to predict the 3D gaze direction vector up
to 50 ms into the future, given a sequence of previous eye images. For a dataset recorded
at 100 Hz, 50 ms is equivalent to 5 frames. Participants of the challenge were scored on a
test set, which contained the sequence of previous eye images with hidden ground-truth
vectors, using the performance metric of **_Prediction Error (PE)_**, defined as:

$PE=\frac{PE_t}{5}$, where
$PE_t=\frac{\sum_{s}^Sd(g_{t,s}, \hat{g}_{t, s})}{|S|}$ for $t\in [1, 5]$

where:
* $|S|$ - is the number of sequences in the test set
* $g_{t,s}$ - the ground-truth 3D gaze
direction vector 
* $\hat{g}_{t, s}$ - the corresponding gaze prediction


$d(.)$ - per-frame angular error between
estimated and ground-truth gaze vectors such that:

$d(g, \hat{g})=\arccos{\frac{g \cdot \hat{g}}{\Vert g \Vert \cdot \Vert\hat{g} \Vert} }$

The 3D gaze direction vector is defined as the 3D unit (normalized) vector in the
direction of the visual axis, represented in the 3D Cartesian coordinate system. For training,
participants were provided with both eye images and ground-truth vectors, so that they
could train appearance-based gaze estimation models using the given dataset. No subject
calibration details were provided.


### Dataset structure:
* Eye images and true gaze vectors were provided for training and validation parts
* Test part of the dataset consists only of eye images without any gaze vectors 

### Dataset size:
* train - 128,000 images (100-frame sequences)
* validation - 70,400 images (55-frame sequences)
* test - 352,000 images (55-frame sequences)


[Back to README](../README.md)