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

from sklearn.metrics import accuracy_score

from  models import *

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
        self.data_loader = DataLoader(dataset_name=self.dataset_name,img_res=self.img_shape,use_test_in_batch=True)


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
        self.d = build_discriminator(img_shape=self.img_shape,df=64,num_classes=self.num_classes,act_multi_label='sigmoid')
        print("******** Discriminator/Classifier ********")
        self.d.summary()
        self.d.compile(loss=[
            'binary_crossentropy',  # gan
            'binary_crossentropy'  # 0 - Angry 
            # 'binary_crossentropy'  # 1 - Disgust
            # 'binary_crossentropy'  # 2 - Fear
            # 'binary_crossentropy'  # 3 - Happy 
            # 'binary_crossentropy'  # 4 - Sad 
            # 'binary_crossentropy'  # 5 - Surprise 
            # 'binary_crossentropy'  # 6 - Neutral 
            ],
            optimizer=optimizer,
            metrics=['accuracy'],
            loss_weights=[
            1 , # gan
            1  # 0 - Angry
            # 3 , # 1 - Disgust
            # 3 , # 2 - Fear
            # 3 , # 3 - Happy 
            # 3 , # 4 - Sad
            # 3 , # 5 - Surprise 
            # 3 , # 6 - Neutral 
            ])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        #self.g = self.build_generator()
        self.g_enc , self.g_dec = build_generator_enc_dec(img_shape=(48,48,1),gf=64,num_classes=7,channels=1,tranform_layer=True)
        print("******** Generator_ENC ********")
        self.g_enc.summary()
        print("******** Generator_DEC ********")
        self.g_dec.summary()

        # Input images from both domains
        img = Input(shape=self.img_shape)
        label0 = Input(shape=(self.num_classes,))
        label1 = Input(shape=(self.num_classes,))

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
    
    def generate_new_labels(self,labels0):
        labels1 = [] 
        for i in range(len(labels0)):
            allowed_values = list(range(0, self.num_classes))
            allowed_values.remove(labels0[i])
            labels1.append(random.choice(allowed_values))
        return np.array(labels1,'int32')
    
    def generate_new_labels_all(self,labels0):
        labels_all = [] 
        for i in range(len(labels0)):
            allowed_values = list(range(0, self.num_classes))
            allowed_values.remove(labels0[i])
            labels_all.append(np.array(allowed_values,'int32'))
        return np.array(labels_all,'int32')

    def train(self, epochs, batch_size=1, sample_interval=50 , d_g_ratio=5):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        #valid = np.ones((batch_size,) + self.disc_patch)
        #fake = np.zeros((batch_size,) + self.disc_patch)

        valid = np.ones((batch_size,1) )
        fake = np.zeros((batch_size,1) )

        null_labels = np.zeros((batch_size,7) )

        for epoch in range(epochs):
            for batch_i, (labels0 , imgs) in enumerate(self.data_loader.load_batch(batch_size=batch_size)):
                labels1_all = self.generate_new_labels_all(labels0)
                #labels01 = self.generate_new_labels(labels0)
                #print("labels0:",labels0)
                #print("labels1_all:",labels1_all)
                
                #print("labels1_all[:,0]",labels1_all[:,0])

                labels0_cat = to_categorical(labels0, num_classes=self.num_classes)
                #
                labels1_all_1 = to_categorical(labels1_all[:,0], num_classes=self.num_classes)
                labels1_all_2 = to_categorical(labels1_all[:,1], num_classes=self.num_classes)
                labels1_all_3 = to_categorical(labels1_all[:,2], num_classes=self.num_classes)
                labels1_all_4 = to_categorical(labels1_all[:,3], num_classes=self.num_classes)
                labels1_all_5 = to_categorical(labels1_all[:,4], num_classes=self.num_classes)
                labels1_all_6 = to_categorical(labels1_all[:,5], num_classes=self.num_classes)
                
                #print("labels1_all_1",labels1_all_1)
                
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                #fakes = self.g.predict([labels1,imgs])
                zs1,zs2,zs3,zs4 = self.g_enc.predict(imgs)
                fakes_1 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_1])
                fakes_2 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_2])
                fakes_3 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_3])
                fakes_4 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_4])
                fakes_5 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_5])
                fakes_6 = self.g_dec.predict([zs1,zs2,zs3,zs4,labels1_all_6])
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


                idx = np.random.permutation(7*labels0.shape[0])
                _labels_cat = np.concatenate([labels0_cat,
                                              null_labels,
                                              null_labels,
                                              null_labels,
                                              null_labels,
                                              null_labels,
                                              null_labels])
                _imgs = np.concatenate([imgs,
                                        fakes_1,
                                        fakes_2,
                                        fakes_3,
                                        fakes_4,
                                        fakes_5,
                                        fakes_6])
                _vf = np.concatenate([valid,fake,fake,fake,fake,fake,fake])
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
                    g_loss = self.combined.train_on_batch([imgs, labels0_cat, labels1_all_1],
                                                            [valid, labels1_all_1, imgs])
                    
                    g_loss = self.combined.train_on_batch([imgs, labels0_cat, labels1_all_2],
                                                            [valid, labels1_all_2, imgs])
                    
                    g_loss = self.combined.train_on_batch([imgs, labels0_cat, labels1_all_3],
                                                            [valid, labels1_all_3, imgs])
                    
                    g_loss = self.combined.train_on_batch([imgs, labels0_cat, labels1_all_4],
                                                            [valid, labels1_all_4, imgs])
                    
                    g_loss = self.combined.train_on_batch([imgs, labels0_cat, labels1_all_5],
                                                            [valid, labels1_all_5, imgs])
                    
                    g_loss = self.combined.train_on_batch([imgs, labels0_cat, labels1_all_6],
                                                            [valid, labels1_all_6, imgs])

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
        
        ## disc
        labels0_d , imgs_d = self.data_loader.load_data(batch_size=64, is_testing=True)

        gan_pred_prob,class_pred_prob = self.d.predict(imgs_d)
        
        #print("gan_pred_prob:",gan_pred_prob.shape,gan_pred_prob) #64x1
        gan_pred = (gan_pred_prob > 0.5)*1.0
        gan_pred = gan_pred.reshape((64,))
        #print("gan_pred:",gan_pred.shape,gan_pred)  
        #print("class_pred_prob:",class_pred_prob.shape,class_pred_prob) #64x7 
        class_pred = np.argmax(class_pred_prob,axis=1)
        #print("class_pred:",class_pred.shape,class_pred)  
        #print("labels0_d:",labels0_d)

        gan_test_accuracy = accuracy_score(y_true=np.ones(64), y_pred=gan_pred)
        class_test_accuracy = accuracy_score(y_true=labels0_d, y_pred=class_pred)

        print("*** gan_test_accuracy   :",gan_test_accuracy,"***")
        print("*** class_test_accuracy :",class_test_accuracy,"***")

        ## gen 
        labels0_ , imgs_ = self.data_loader.load_data(batch_size=1, is_testing=True)
        labels1_all = self.generate_new_labels_all(labels0_)

        labels0_cat = to_categorical(labels0_, num_classes=self.num_classes)
        labels1_all_1 = to_categorical(labels1_all[:,0], num_classes=self.num_classes)
        labels1_all_2 = to_categorical(labels1_all[:,1], num_classes=self.num_classes)
        labels1_all_3 = to_categorical(labels1_all[:,2], num_classes=self.num_classes)
        labels1_all_4 = to_categorical(labels1_all[:,3], num_classes=self.num_classes)
        labels1_all_5 = to_categorical(labels1_all[:,4], num_classes=self.num_classes)
        labels1_all_6 = to_categorical(labels1_all[:,5], num_classes=self.num_classes)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        #fake_ = self.g.predict([labels1_,imgs_])
        zs1_,zs2_,zs3_,zs4_ = self.g_enc.predict(imgs_)
        fake_1 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_1])
        fake_2 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_2])
        fake_3 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_3])
        fake_4 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_4])
        fake_5 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_5])
        fake_6 = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels1_all_6])
      
        # Translate back to original domain
        reconstr_ = self.g_dec.predict([zs1_,zs2_,zs3_,zs4_,labels0_cat])

        gen_imgs = np.concatenate([imgs_, 
                                   fake_1, 
                                   fake_2, 
                                   fake_3, 
                                   fake_4, 
                                   fake_5, 
                                   fake_6,
                                   reconstr_])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Orig:'+str(self.lab_dict[labels0_.item(0)]), 
                  'Trans:'+str(self.lab_dict[labels1_all[:,0].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,1].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,2].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,3].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,4].item(0)]),
                  'Trans:'+str(self.lab_dict[labels1_all[:,5].item(0)]),
                  'Reconstr.']
        r, c = 2, 4
        fig, axs = plt.subplots(r, c)
        

        if not os.path.exists( "images/%s/"% (self.dataset_name)):
            os.makedirs( "images/%s/"% (self.dataset_name)  )
        
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt].reshape((self.img_rows,self.img_cols)),cmap='gray')
                axs[i,j].set_title(titles[cnt])
                axs[i,j].axis('off')
                cnt += 1
                
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = CCycleGAN()
    gan.train(epochs=600, batch_size=64, sample_interval=200 , d_g_ratio=1)
