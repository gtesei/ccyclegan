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
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
import numpy as np
import os
import random 

import tensorflow as tf 

from keras.utils import to_categorical

class CCycleGAN():
    def __init__(self,img_rows = 48,img_cols = 48,channels = 1, num_classes=7, latent_dim=100):
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
        self.data_loader = DataLoader(dataset_name=self.dataset_name,img_res=self.img_shape)


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 1                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = Adam(0.0002, 0.5) 

        # Build and compile the discriminators
        self.d = self.build_discriminator()
        print("******** Discriminator/Classifier ********")
        self.d.summary()
        self.d.compile(loss=['binary_crossentropy','categorical_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'],
            loss_weights=[1, 1])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        #self.g = self.build_generator()
        self.g_enc , self.g_dec = self.build_generator_enc_dec()
        print("******** Generator_ENC ********")
        self.g_enc.summary()
        print("******** Generator_DEC ********")
        self.g_dec.summary()

        # Input images from both domains
        img = Input(shape=self.img_shape)
        label0 = Input(shape=(1,))
        label1 = Input(shape=(1,))

        # Translate images to the other domain
        #fake = self.g([label1,img])
        z1,z2,z3,z4 = self.g_enc(img)
        fake = self.g_dec([z1,z2,z3,z4,label1])
        # Translate images back to original domain
        #reconstr = self.g([label0,fake])
        reconstr = self.g_dec([z1,z2,z3,z4,label0])
        # Identity mapping of images
        #img_id = self.g([label0,img])

        # For the combined model we will only train the generators
        self.d.trainable = False

        # Discriminators determines validity of translated images gan_prob,class_prob [label,img], [gan_prob,class_prob]
        gan_valid , class_valid = self.d(fake)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img,label0,label1],
                              outputs=[ gan_valid, class_valid, 
                                        reconstr])
        self.combined.compile(loss=['binary_crossentropy','categorical_crossentropy',
                                    'mae'],
                            loss_weights=[  1 , 1, 1],
                            optimizer=optimizer)

    def build_generator_enc_dec(self):
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

        G_enc = Model(img,[d1,d2,d3,d4])

        #### 
        
        d1_ = Input(shape=(24, 24, 32))
        d2_ = Input(shape=(12, 12, 64))
        d3_ = Input(shape=(6, 6, 128))
        d4_ = Input(shape=(3, 3, 256))

        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 72)(label))
        label_embedding = Reshape((3,3,8))(label_embedding)
        #flat_img = Flatten()(d4_)
        d44 = concatenate([d4_, label_embedding],axis=-1)
        #d44 = multiply([flat_img, label_embedding])
        d444 = Reshape((3,3,264))(d44)

        ####

        # Upsampling
        u1 = deconv2d(d444, d3_, self.gf*4)
        u2 = deconv2d(u1, d2_, self.gf*2)
        u3 = deconv2d(u2, d1_, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        G_dec = Model([d1_,d2_,d3_,d4_,label],output_img)

        return G_enc , G_dec


    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)
        
        #label = Input(shape=(1,), dtype='int32')
        #label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        #flat_img = Flatten()(img)
        #model_input = multiply([flat_img, label_embedding])
        #d0 = Reshape(self.img_shape)(model_input)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        flat_repr = Flatten()(d4)

        #validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        print("flat_repr.get_shape().as_list():",flat_repr.get_shape().as_list())
        print("flat_repr.get_shape().as_list()[1:]:",flat_repr.get_shape().as_list()[1:])

        gan_logit = Dense(800)(flat_repr)
        gan_logit = LeakyReLU(alpha=0.2)(gan_logit)
        gan_prob = Dense(1, activation='sigmoid')(gan_logit)


        class_logit = Dense(800)(flat_repr)
        class_logit = LeakyReLU(alpha=0.2)(class_logit)
        class_prob = Dense(self.num_classes, activation='softmax')(class_logit)


        #### 
        #label = Input(shape=(1,), dtype='int32')
        #label_embedding = Flatten()(Embedding(self.num_classes, 9)(label))
        #flat_img = Flatten()(validity)
        #d44 = multiply([flat_img, label_embedding])
        #d444 = Reshape(validity.get_shape().as_list()[1:])(d44)
        ####

        return Model(img, [gan_prob,class_prob])
    
    def generate_new_labels(self,labels0):
        labels1 = [] 
        for i in range(len(labels0)):
            allowed_values = list(range(0, self.num_classes))
            allowed_values.remove(labels0[i])
            labels1.append(random.choice(allowed_values))
        return np.array(labels1,'int32')

    def train(self, epochs, batch_size=1, sample_interval=50 , d_g_ratio=5):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        #valid = np.ones((batch_size,) + self.disc_patch)
        #fake = np.zeros((batch_size,) + self.disc_patch)

        valid = np.ones((batch_size,1) )
        fake = np.zeros((batch_size,1) )

        for epoch in range(epochs):
            for batch_i, (labels0 , imgs) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):
                labels1 = self.generate_new_labels(labels0)
                labels01 = self.generate_new_labels(labels0)

                labels0_cat = to_categorical(labels0, num_classes=self.num_classes)
                labels01_cat = to_categorical(labels01, num_classes=self.num_classes)
                
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                #fakes = self.g.predict([labels1,imgs])
                zs1,zs2,zs3,zs4 = self.g_enc.predict(imgs)
                fakes = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1])
                #print("fake",str(fake.shape))

                # Train the discriminators (original images = real / translated = Fake)
                #d_loss_real = self.d.train_on_batch([labels0,imgs], valid)
                #d_loss_real_fake = self.d.train_on_batch([labels01,imgs], fake)
                #d_loss_fake = self.d.train_on_batch([labels1,fakes], fake)
                #d_loss = (1/2) * np.add(d_loss_real, d_loss_fake)

                #d_loss_real = self.d.train_on_batch(imgs, [valid,labels0_cat])
                #d_loss_fake  = self.d.train_on_batch(fakes, [fake,labels01_cat])

                #print("d_loss_real:",d_loss_real)
                #print("d_loss_fake:",d_loss_fake)

                #d_loss_gan = (1/2) * np.add(d_loss_real[0], d_loss_fake[0])
                #d_loss_class = (1/2) * np.add(d_loss_real[1], d_loss_fake[1])


                idx = np.random.permutation(2*labels1.shape[0])
                _labels_cat = np.concatenate([labels0_cat,labels01_cat])
                _imgs = np.concatenate([imgs,fakes])
                _vf = np.concatenate([valid,fake])
                _labels_cat = _labels_cat[idx]
                _imgs = _imgs[idx]
                _vf = _vf[idx]

                d_loss  = self.d.train_on_batch(_imgs, [_vf,_labels_cat])

                #d_loss = self.d.train_on_batch(_imgs, [_vf,_labels])

                if batch_i % d_g_ratio == 0:

                    # ------------------
                    #  Train Generators
                    # ------------------

                    # Train the generators
                    g_loss = self.combined.train_on_batch([imgs, labels0, labels1],
                                                            [valid, labels01_cat, imgs])

                    elapsed_time = datetime.datetime.now() - start_time

                    # Plot the progress
                    # print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                    #                                                         % ( epoch, epochs,
                    #                                                             batch_i, self.data_loader.n_batches,
                    #                                                             d_loss[0], 100*d_loss[1],
                    #                                                             g_loss[0],
                    #                                                             np.mean(g_loss[1:2]),
                    #                                                             np.mean(g_loss[2:3]),
                    #                                                             np.mean(g_loss[3:4]),
                    #                                                             elapsed_time))
                    # print("[Epoch %d/%d]"% ( epoch, epochs), 
                    #     "[d_loss_gan",d_loss_gan,"]", 
                    #     "[d_loss_class:", d_loss_class,"]", 
                    #     "[g_loss_gan:",g_loss[0],"]", 
                    #     "[g_loss_class:", g_loss[1],"]", 
                    #     "[recon_loss:",g_loss[2],"]",
                    #     "[time:",elapsed_time,"]")

                    print ("[Epoch %d/%d] [Batch %d/%d] [D_gan loss: %f, acc_gan: %3d%%] [D_cl loss: %f, acc_cl: %3d%%] [G_gan loss: %05f, G_cl: %05f, recon: %05f] time: %s " \
                        % ( epoch, epochs,
                            batch_i, self.data_loader.n_batches,
                            d_loss[1],100*d_loss[3],d_loss[2],100*d_loss[4],
                            g_loss[1],g_loss[2],g_loss[3],
                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    
        

    def sample_images(self, epoch, batch_i):
        #os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 1, 3

        labels0_ , imgs_ = self.data_loader.load_data(batch_size=1, is_testing=True)
        labels1_ = self.generate_new_labels(labels0_)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        #fake_ = self.g.predict([labels1_,imgs_])
        zs1_,zs2_,zs3_,zs4_ = self.g_enc.predict(imgs_)
        fake_ = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_])
      
        # Translate back to original domain
        reconstr_ = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels0_])

        gen_imgs = np.concatenate([imgs_, fake_, reconstr_])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Orig-l0:'+str(self.lab_dict[labels0_.item(0)]), 'Trans-l1:'+str(self.lab_dict[labels1_.item(0)]), 'Reconstr.']
        fig, axs = plt.subplots(r, c)
        cnt = 0

        if not os.path.exists( "images/%s/"% (self.dataset_name)):
            os.makedirs( "images/%s/"% (self.dataset_name)  )
        for j in range(c):
            axs[j].imshow(gen_imgs[cnt].reshape((self.img_rows,self.img_cols)),cmap='gray')
            axs[j].set_title(titles[j])
            axs[j].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = CCycleGAN()
    gan.train(epochs=400, batch_size=64, sample_interval=200 , d_g_ratio=10)
