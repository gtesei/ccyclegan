# Conditional Cycle-Consistent GAN
Generative adversarial networks has been widely explored for generating realist images but their capabilities in multimodal image-to-image translations where a single input may correspond to many possible outputs in a conditional generative model setting have been vaguely explored (Zhu et al., 2017). Moreover, applying such capabilities of GANs in the context of facial expression generation, where even relatively little unnatural distortions of generated images can be easily detectable by a human, to my knowledge, is a green field. Thus, the novelty of this study consists in experimenting the synthesis of facial expressions, i.e. learning to translate an image from a domain X (e.g. the face image of a person) conditioned on a given facial expression label (e.g. “joy”) to the same domain X but conditioned on a different facial expression label (e.g. “surprise”).

## Installation
    $ git https://github.com/gtesei/ccyclegan.git
    $ cd ccyclegan/
    $ sudo pip3 install -r requirements.txt
    





