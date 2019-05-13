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
T1 | ccyclegan_t1.py | Baseline - LGAN is replaced by the negative log likelihood objective by a least-squares loss [X. Mao, Q. Li, H. Xie, R. Y. Lau, Z. Wang, and S. P. Smolley. Least squares generative adversarial networks. In CVPR. IEEE, 2017]. Also, we
adopt the technique of [Y. Taigman, A. Polyak, and L. Wolf. Unsupervised cross-domain image generation. In ICLR, 2017.] and regularize the generator to be near an identity mapping when real samples of the target domain are provided as the input to the generator. Weights are the same of the paper of CycleGAN, i.e. Identity loss = 0.1*Cycle-consistency loss| The generator has a loss too high vs. discriminator |
--- | --- | --- | --- |
--- | --- | --- | --- |
--- | --- | --- | --- |
--- | --- | --- | --- |
--- | --- | --- | --- |








    





