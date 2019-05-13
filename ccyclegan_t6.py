from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import Reshape
from keras.layers.merge import concatenate
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import random 
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


class CCycleGAN():
    def __init__(self,img_rows = 48,img_cols = 48,channels = 3, num_classes=7, latent_dim=99):
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        ## dict
        self.lab_dict = {0: "Angry", 1: "Disgust" , 2: "Fear" , 3: "Happy" , 4: "Sad" , 5: "Surprise" , 6: "Neutral"}

        # Configure data loader
        self.dataset_name = 'fer2013'
        self.data_loader = DataLoader(dataset_name=self.dataset_name)


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d = self.build_discriminator2()
        print("******** Discriminator ********")
        self.d.summary()
        self.d.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g = self.build_generator()
        print("******** Generator ********")
        self.g.summary()

        # Input images from both domains
        img = Input(shape=self.img_shape)
        label0 = Input(shape=(1,))
        label1 = Input(shape=(1,))

        # Translate images to the other domain
        fake = self.g([label1,img])
        # Translate images back to original domain
        reconstr = self.g([label0,fake])
        # Identity mapping of images
        #img_id = self.g([label0,img])

        # For the combined model we will only train the generators
        self.d.trainable = False

        # Discriminators determines validity of translated images
        valid = self.d([label1,fake])

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img,label0,label1],
                              outputs=[ valid, 
                                        reconstr])
        self.combined.compile(loss=['mse',
                                    'mae'],
                            loss_weights=[  1, 
                                            self.lambda_cycle],
                            optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        img = Input(shape=self.img_shape)
        
        # Downsampling
        d1 = conv2d(img, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        
        ###
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        
        latent_vect = Flatten()(d4)
        latent_concat = concatenate([latent_vect, label_embedding])
        
        d5 = Reshape((3, 3, -1))(latent_concat)
        ###

        # Upsampling
        u1 = deconv2d(d5, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model([label,img], output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)
        
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        
        flat_img = Flatten()(img)
        
        model_input = multiply([flat_img, label_embedding])
        d0 = Reshape(self.img_shape)(model_input)

        d1 = d_layer(d0, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([label,img], validity)
    
    def build_discriminator2(self):

        #img = Input(shape=self.img_shape)
        
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))
        
        #flat_img = Flatten()(img)   
        
        #model_input = multiply([flat_img, label_embedding])
        
        base_model  = ResNet50(weights= 'imagenet', include_top=False, input_shape= (48,48,3))
        
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        #latent_vect = Flatten()(x)
        latent_concat = concatenate([x, label_embedding])
        # let's add a fully-connected layer
        f = Dense(1024, activation='relu')(latent_concat)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(1, activation='sigmoid')(f)
        
        return Model([label,base_model.input], predictions)
    
    def generate_new_labels(self,labels0):
        labels1 = [] 
        for i in range(len(labels0)):
            allowed_values = list(range(0, self.num_classes))
            allowed_values.remove(labels0[i])
            labels1.append(random.choice(allowed_values))
        return np.array(labels1,'int32')

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        #valid = np.ones((batch_size,) + self.disc_patch)
        #fake = np.zeros((batch_size,) + self.disc_patch)'
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        for epoch in range(epochs):
            for batch_i, (labels0 , imgs) in enumerate(self.data_loader.load_batch(batch_size=batch_size,convertRGB=True)):
                labels1 = self.generate_new_labels(labels0)
                labels01 = self.generate_new_labels(labels0)
                
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fakes = self.g.predict([labels1,imgs])
                #print("fake",str(fake.shape))
                
                idx = np.random.permutation(3*labels1.shape[0])
                _labels = np.concatenate([labels0,labels01,labels1])
                _imgs = np.concatenate([imgs,imgs,fakes])
                _vf = np.concatenate([valid,fake,fake])

                _labels = _labels[idx]
                _imgs = _imgs[idx]
                _vf = _vf[idx]

                # Train the discriminators (original images = real / translated = Fake)
#                d_loss_real = self.d.train_on_batch([labels0,imgs], valid)
#                d_loss_real_fake = self.d.train_on_batch([labels01,imgs], fake)
#                d_loss_fake = self.d.train_on_batch([labels1,fakes], fake)
#                d_loss = (1/3) * np.add(np.add(d_loss_real, d_loss_real_fake) ,d_loss_fake)
                
                d_loss = self.d.train_on_batch([_labels,_imgs], _vf)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs, labels0, labels1],
                                                        [valid,
                                                        imgs])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:2]),
                                                                            np.mean(g_loss[2:3]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    
        

    def sample_images(self, epoch, batch_i):
        #os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 1, 3

        labels0_ , imgs_ = self.data_loader.load_data(batch_size=1, is_testing=True, convertRGB=True)
        labels1_ = self.generate_new_labels(labels0_)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_ = self.g.predict([labels1_,imgs_])
      
        # Translate back to original domain
        reconstr_ = self.g.predict([labels0_,fake_])

        gen_imgs = np.concatenate([imgs_, fake_, reconstr_])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Orig-l0:'+str(self.lab_dict[labels0_.item(0)]), 'Trans-l1:'+str(self.lab_dict[labels1_.item(0)]), 'Reconstr.']
        fig, axs = plt.subplots(r, c)
        cnt = 0

        if not os.path.exists( "images/%s/"% (self.dataset_name)):
            os.makedirs( "images/%s/"% (self.dataset_name)  )
        for j in range(c):
            axs[j].imshow(gen_imgs[cnt].squeeze())
            axs[j].set_title(titles[j])
            axs[j].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = CCycleGAN()
    gan.train(epochs=200, batch_size=64, sample_interval=200)
