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

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.utils import np_utils


class CCycleGAN():
    def __init__(self,img_rows = 48,img_cols = 48,channels = 1, num_classes=7, latent_dim=99,PREFIX='saved_model/'):
        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.PREFIX=PREFIX
        
        ## dict
        self.lab_dict = {0: "Angry", 1: "Disgust" , 2: "Fear" , 3: "Happy" , 4: "Sad" , 5: "Surprise" , 6: "Neutral"}

        # Configure data loader
        self.dataset_name = 'fer2013'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,img_res=self.img_shape)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.d = self.build_discriminator2()
        print("******** Discriminator ********")
        self.d.summary()
        self.d.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    
    
    def build_discriminator2(self):
        
        base_model  = ResNet50(weights= 'imagenet', include_top=False, input_shape= (48,48,3))
        
        # add a global spatial average pooling layer
        x = base_model.output
        latent_repr = GlobalAveragePooling2D()(x)
        
        # let's add a fully-connected layer
        f = Dense(1024, activation='relu')(latent_repr)
        predictions = Dense(self.num_classes, activation='softmax')(f)
        
        return Model(base_model.input, predictions)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        
        earlystopper = EarlyStopping(patience=20, verbose=1,monitor='val_acc',mode='max')
        checkpointer = ModelCheckpoint(self.PREFIX+'classifier_2.h5', verbose=1, save_best_only=True,monitor='val_acc',mode='max')
        reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1,monitor='val_acc',mode='max')
        results = self.d.fit(self.data_loader.img_vect_train_RGB, 
                            np_utils.to_categorical(self.data_loader.lab_vect_train,num_classes=self.num_classes),
                    validation_data=[self.data_loader.img_vect_test_RGB,
                                     np_utils.to_categorical(self.data_loader.lab_vect_test,num_classes=self.num_classes)],
                    batch_size=batch_size, epochs=epochs,
                    callbacks=[earlystopper, checkpointer,reduce_lr], shuffle=True)
        
if __name__ == '__main__':
    gan = CCycleGAN()
    gan.train(epochs=200, batch_size=64, sample_interval=200)
