# Low Light Enhancement Using Zero-Reference Deep Curve Estimation
## Introduction
Most photos, especially with the growth of mobile cameras, captured suffer compromised aesthetic
quality because of incorrect or not optimal lighting conditions. These sub-optimal conditions
are mostly because of the lack of awareness about lighting strategy, which is the case with nonprofessional
or laymen photographer. To overcome this, we can process these low light images with
low light enhancing techniques that will improve the quality of the original picture. Here,
I explored one such technique from [[1]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf). The method explored here, formulates light enhancement
as a task of image specific curve estimations using a light weight deep neural network.

## Methodology
**Light Enhancement Curves:** Light enhancement curves are
inspired from curves that are frequently used in photo editing to adjust the pictures, by operating
at pixel level. They try to map low light image to its enhanced image automatically. The neural
model, we will discuss, essentially tries to model these light enhancement curves, given the input
image. How are these curves used to enhance the image though? \
Below equation mathematically
answers this question\
$$LE_n(x) = LE_{n−1}(x) + A_n(x)LE_{n−1}(x)(1 − LE_{n−1}(x))$$\
Where $A_n$ is curve map that does pixel wise curving.
It is very apparent that this equation is recursive. Meaning to obtain nth level enhanced image it
needs ${n − 1}^{th}$ enhanced image and nth pixel level curve map. Now the goal in this work was to estimate the $A$′s,
for which a deep learning architecture, called zero-reference DCE (DCE-Net), is used.\
\
**DCE-Net:**
The models learns to map the input, typically a low light image, to its best
fitting curve parameter map. The architecture is a plain CNN assemble of seven convolution layers
with symmetrical concatenation. Each layer consists of 32 convolutional kernels of size 3 × 3 and
stride 1 followed by ReLU activation. No down sampling or batch normalization is done since they
break the relation among the neighbouring pixels. Below is the visual representation of the model
architecture.

1) Training: Look for the training script train_model.py
2) Inference: Look for the inferency script inference.py
3) Most of these scripts can be run in other environments with few tweaks
4) For the dataset refer to here https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view
5) Find the stable model under models

References\
[1] Guo, Chunle, et al. "Zero-reference deep curve estimation for low-light image enhancement." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.\
[2] \href{https://keras.io/examples/vision/zero_dce/}{https://keras.io/examples/vision/zero\_dce/}

