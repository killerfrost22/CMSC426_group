# CMSC426_group 2021 Fall Repository
## Authors: 
Yizhan Ao
Yingqiao Gou
### Project 1
Color Segmentation using GMM
color segmentation, bayer filter, image acquisition, color-space
#### Estimation
Estimating p(Cl|x) directly is too difficult. Luckily, we have Bayes rule to rescue us! Bayes rule applied onto p(Cl|x) gives us the following:
p(Cl|x)=p(x|Cl)p(Cl)∑li=1p(x|Ci)p(Ci)
p(Cl|x) is the conditional probability of a color label given the color observation and is called the Posterior. p(x|Cl) is the conditional probability of color observation given the color label and is generally called the Likelihood. 
#### Color Classification using a Gaussian Mixture Model (GMM)
In this case, one has to come up with a wierd looking fancy function to bound the color which is generally mathematically very difficult and computationally very expensive. 
### Project 2 Panorama Stitching
Stitching multiple images seemlessly to create a panorama
### Project 3 Rotobrush
local classifiers, color confidence, shape confidence, local boundary deformation
#### Segmenting with Localized Classifiers
##### Initializing the Color Model and Color Model Confidence
To initialize the color model, we followed the process in Video Snap Cut and converted the input
image to Lab color space.
##### Updating Local Windows
To estimate the large amounts of motion in the object, we used detectSURFFeatures, which is
rotational invariant. We tried to force matching to focus on the foreground by removing the
background (setting pixels to NaN) but this often resulted in the algorithm not finding enough
matching points.
#### Estimating the Local Boundary Deformation
To track small bits of motion, the transformation alone was not enough. We used the optical flow
to account for these small changes.
#### Updating Local Classifier
We first convert the image to Lab color space. For each window, we calculated the probability a
pixel is foreground or background using pdf of the old GMMs and the current image.
### Project 4 SfM or SLAM
Structure from Motion (SfM) or Simultaneous Localization and Mapping (SLAM)
