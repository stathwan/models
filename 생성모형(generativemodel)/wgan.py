
import os
#os.chdir('c:/users/2017B221/desktop/othermodel/wgan')
import glob

#from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import cv2
import keras.backend as K
import numpy as np
        
import sys
import numpy as np
import os

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

class WGAN():
    def __init__(self):
        self.img_rows = 48
        self.img_cols = 48
        self.channels = 3

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        noise_shape = (100,)
        
        model= Sequential()
        model.add(Dense(1024*3*3, activation='relu' ,input_shape=(noise_shape)))
        model.add(Reshape((3, 3, 1024)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(512,activation='relu' ,kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(256,activation='relu' ,kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128,activation='relu' ,kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels,activation='tanh', kernel_size=3, padding="same"))
        
#        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape))
        model.add(LeakyReLU(alpha=0.2))   
        model.add(Dropout(0.70))
        model.add(Conv2D(128, kernel_size=3, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.70))
        model.add(Conv2D(256, kernel_size=3, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.70))
        model.add(Conv2D(512, kernel_size=3, strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
#        model.summary()
        img = Input(shape=img_shape)
        features = model(img)
        valid = Dense(1, activation="linear")(features)

        return Model(img, valid)
        
    def load_data(self):
        impath=glob.glob('./data/*')
        img_shape = (len(impath),self.img_rows, self.img_cols, self.channels)
        trips_img=np.empty(shape=img_shape)

        for i in range(int(len(impath))) :
            img=cv2.imread(impath[i])
            img=np.expand_dims(img,axis=0)
            if i ==0 :
                trips_img= img
            else:   
                trips_img=np.concatenate((trips_img,img))
                # print(str(i+1),"번째 이미지가 합쳐졌습니다.")  
        return(trips_img.astype('float32')) 
        
    def load_model(self,path1,path2):
         self.generator.load_weights(path1)
         self.discriminator.load_weights(path2)
        
    def train(self, epochs, batch_size=128, sample_interval=50):
        self.generator.load_weights('./saved_model/generator16000.hdf5')
        self.discriminator.load_weights('./saved_model/discriminator16000.hdf5')
        # Load the dataset
        X_train = self.load_data()
        print(X_train.shape)
        # Rescale -1 to 1
        X_train = ( X_train.astype(np.float32) -127.5) / 127.5

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]

                noise = np.random.normal(0, 1, (half_batch, 100))

                # Generate a half batch of new images
                gen_imgs = self.generator.predict(noise)
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, -np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip discriminator weights
                for l in self.discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, -np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch % 1000 == 0 :
                self.generator.save_weights('./saved_model/generator{}.hdf5'.format(epoch))
                self.discriminator.save_weights('./saved_model/discriminator{}.hdf5'.format(epoch))
                
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        digit_size=48
        figure = np.zeros((digit_size * r, digit_size * c,3))
        k=0
        for i in range(r):
            for j in range(c):
                digit = gen_imgs[k].reshape(digit_size, digit_size,3)
                figure[i * digit_size: (i + 1) * digit_size,j * digit_size: (j + 1) * digit_size,0:3] = digit
                k+=1
#        cv2.imsave('images/mnist_%d.png',figure[...,[2,1,0]])
        plt.imsave('images/mnist_{}.jpg'.format(epoch),figure[...,[2,1,0]])
        
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)

wgan = WGAN()
#wgan.load_model('./saved_model/generator38000.hdf5','./saved_model/discriminator38000.hdf5')
wgan.train(epochs=10000000000000, batch_size=32, sample_interval=100)

