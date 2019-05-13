# Conditional Cycle-Consistent GAN
Generative adversarial networks has been widely explored for generating realist images but their capabilities in multimodal image-to-image translations where a single input may correspond to many possible outputs in a conditional generative model setting have been vaguely explored. Moreover, applying such capabilities of GANs in the context of facial expression generation, where even relatively little unnatural distortions of generated images can be easily detectable by a human, to my knowledge, is a green field. Thus, the novelty of this study consists in experimenting the synthesis of facial expressions, i.e. learning to translate an image from a domain X (e.g. the face image of a person) conditioned on a given facial expression label (e.g. “joy”) to the same domain X but conditioned on a different facial expression label (e.g. “surprise”).

## Installation
    $ git https://github.com/gtesei/ccyclegan.git
    $ cd ccyclegan/
    $ sudo pip3 install -r requirements.txt
    
## Dataset 
FER2013 consists of 28,709 48x48 pixel grayscale images of faces annotated with the emotion of facial expression as one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.
You need to download the dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) and put __fer2013.csv__ under the folder __datasets__. 

<img src="images/fer2013_sample.png" align="middle" /> 

## Experiment Log

Id | Code | Description | Notes | 
--- | --- | --- | --- |
T1 | ccyclegan_t1.py | Baseline - GAN loss (the negative log likelihood objective) is replaced by a least-squares loss [X. Mao, Q. Li, H. Xie, R. Y. Lau, Z. Wang, and S. P. Smolley. _Least squares generative adversarial networks._ In CVPR. IEEE, 2017]. Also, we adopt the technique of [Y. Taigman, A. Polyak, and L. Wolf. _Unsupervised cross-domain image generation._ In ICLR, 2017.] and regularize the generator to be near an identity mapping when real samples of the target domain are provided as the input to the generator. Weights are the same of the paper of CycleGAN, i.e. Identity loss weight = 0.1*Cycle-consistency loss weight , G-loss = 1. | G-loss too high compared to D-loss. Let's try to increment the weight of G-loss. |
T2 | ccyclegan_t2.py | Like T1 but the G-loss weight is is set to 20 | No reconstruction in 200 epochs – discriminator has 100% accuracy  |
T3 | ccyclegan_t3.py | Like T1 but Identity loss weight is set to 0, G-loss is binary cross-entropy instead of least-squares loss  (and weight is set to 7). | G-loss too high vs. D-loss |
T4 | ccyclegan_t4.py | Let's simplify the problem: only from domain “Neutral” to domain “Happy”, and from domain “Happy” to domain “Neutral”. No other transformations. | Discriminator ~100% accuracy. This can be due to the fact that, reducing the problem in this way, also the training data is reduced and the generator does not benefit from this. This is an example of situation when Multi-task learning should be applied. Let's restore the problem to its original terms! |
T5 | ccyclegan_t5.py | Like T1 but we concatenate the label encoded after the convolutions as shown in this paper: https://arxiv.org/ftp/arxiv/papers/1708/1708.09126.pdf. Identity loss is removed. | G-loss too high vs. D-loss |
C1 | classifier.py | D of ccyclegan_t4.py | accuracy-train ~ 100%, accuracy-test ~70% which is compatible with the model winner of the Kaggle competition, i.e. 0.71161 |
T6 | ccyclegan_t6.py | Let's make G predict on train images to see if G is able to generate realistic images. | Images not very realistic. |
C2 | classifier2.py | ResNet50 pre-trained on ImageNet (RGG-images). Note: images from grayscale are transformed into RGB. | After 15 epochs we have accuracy-train ~ 95% and accuracy-test ~ 80%, which better than the winner of the Kaggle competition (ResNet was released in 2015 vs. the competition was organized in 2013) |
T7 | ccyclegan_t7.py | Let's use ResNet50 pre-trained on ImageNet as D. | Generated images not very realistic and D-accuracy~100% |
T8 | ccyclegan_t8.py | Change the training procedure removing shuffling and training D only with real|y vs. fake|y images, i.e. removing real|y' samples. | Discriminator has ~100% accuracy on training set, reconstruction loss is low and G-loss is high compared with D-loss  |
T9 | ccyclegan_t9.py | Let's use ResNet50 pre-trained on ImageNet as D and let's freeze it. | Generated images not very realistic and reconstruction loss is low. |










    





