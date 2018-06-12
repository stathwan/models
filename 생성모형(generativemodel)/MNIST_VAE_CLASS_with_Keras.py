# -*- coding: utf-8 -*-
"""
VAE(Variational Autoencoder)입니다. 
주석을 한국어로 작성하여 한국어 사용자들이 이해하기 쉽게 하였습니다.
궁금한점이나/수정이 필요한 부분은 알려주세요
이메일: stat.donghwan@gmail.com
"""

# import module, 모듈 불러오기 
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dropout, BatchNormalization, Dense, Conv2D, Input, Flatten, Reshape, UpSampling2D
from keras.models import Model 
from keras import backend as K 
from keras.engine.topology import Layer
from keras import metrics
from keras.datasets import mnist
from keras.optimizers import adam

# hyperparameter, 초모수설정
K.set_image_data_format('channels_first')


# 필요한 Layer 만들기 
class sampling(Layer):
    
    def __init__(self, latent_size=2,epsilon_stddev= 1.0, **kwargs):
        self.latent_size = latent_size
        self.epsilon_stddev = epsilon_stddev
        super().__init__(**kwargs)
        
    def call(self, theta):
        #값 가져오기
        _mu_hat, _log_var_hat = theta
        _latent_size = self.latent_size
        _batch_size = K.shape(_mu_hat)[0]
        _epsilon_stddev = self.epsilon_stddev
        
        #reparameterization trick
        _epsilon= K.random_normal(shape=(_batch_size,_latent_size), mean=0., stddev= _epsilon_stddev)
        _z=_mu_hat+K.exp(_log_var_hat / 2)*_epsilon
        return  _z   
        
    #아웃풋의 shape정보를 넣어야 한다. 
    def compute_output_shape(self, inshape):
        return (inshape[0])

class VAELoss():
    '''
    참고: sho Tatsuno, VAE ppt, 역 김홍배
    loss = - KL Divergence + Reconstruction Error
    D_KL=(q(z|x)~N(mu, sigma) || p(z)~N(0,I))
        = - 1/2*sum(1+log(sigma^2)-mu^2 + sigma^2)
        
    R.Error \SimEq 1/L*sum(log(p(x|z)))
    if image pixel range is 0 -1, suppose log(p) follow Bernoulli dist.
    log(p(x|z)) = sum(x*log(y) + (1-x)* log(1-y))
    
    '''
    def __init__(self,input_shape,mu_hat,log_variance_hat):
        self.mu_hat = mu_hat
        self.log_variance_hat = log_variance_hat
        self.input_shape = input_shape

        
    def convloss(self,x, x_hat):
        _x= K.flatten(x)
        _y= K.flatten(x_hat)
        _dim=  self.input_shape[1]*self.input_shape[2]
        ReconstructionError= _dim*metrics.binary_crossentropy(_x, _y)
        KLDivergence = 0.5*K.sum(1.+ self.log_variance_hat - K.square(self.mu_hat) - K.exp(self.log_variance_hat), axis=-1)
        loss = K.mean(-KLDivergence + ReconstructionError)
        
        return loss

class VAE():
    
    def __init__(self):
        self.input_shape=(1,28,28)
        self.latent_size=2 # z dim
        self.batch_size=64
        self.epsilon_stddev= 1.0
        self.epochs=5

        
        
        # latent z, 잠재변수 z  
        x = Input(self.input_shape)
        mu_hat,log_variance_hat = self.encoder()(x)
        
        latent_z=sampling(self.latent_size,self.epsilon_stddev)([mu_hat,log_variance_hat])
        y_hat=self.decoder()(latent_z)
        
        self.dcvae=Model(x, y_hat)
        #compile && fit
        self.vaeloss=VAELoss(self.input_shape,mu_hat,log_variance_hat).convloss
        self.dcvae.compile(loss=self.vaeloss, optimizer=adam(0.0002,0.5))
        self.dcvae.summary()
        

        
    #encoder, 인코더        
    def encoder(self):
        x= Input(self.input_shape)
        encoder_h= Conv2D(32 , kernel_size=(3,3), activation='relu')(x)
        encoder_h= Dropout(rate=(0.4))(encoder_h)
        encoder_h= Conv2D(64, kernel_size=(3,3),strides=(2,2), activation='relu')(encoder_h)
        encoder_h= Dropout(rate=(0.4))(encoder_h)
        encoder_h= Conv2D(128, kernel_size=(3,3),strides=(2,2), activation='relu')(encoder_h)
        encoder_h= Flatten()(encoder_h)
        encoder_h= Dense(256)(encoder_h)
    
        
        mu_hat= Dense(self.latent_size,activation='linear')(encoder_h) # default latent_size=2
        # VAE Loss function has log(variance) such that variance should be >0. To make it calculable We train log_variance  
        log_variance_hat= Dense(self.latent_size, activation='linear')(encoder_h) 
        
        encoder=Model(x,[mu_hat,log_variance_hat])
        return encoder
    
        
    def decoder(self):
        #decoder, 디코더
        z= Input((self.latent_size,))
        decoder_h=Dense(7*7*128, activation='relu', name='decoder_h1')(z) # to make generator
        decoder_h=Reshape((128,7,7), input_shape=(7*7*128,), name='decoder_h2')(decoder_h)
        decoder_h=BatchNormalization(name='decoder_h3')(decoder_h)
        decoder_h=Conv2D(64 , kernel_size=(3,3),  activation='relu', padding='same', name='decoder_h4')(decoder_h)
        decoder_h=BatchNormalization(name='decoder_h5')(decoder_h)
        decoder_h=UpSampling2D(name='decoder_h6')(decoder_h)
        decoder_h=Conv2D(32 , kernel_size=(3,3),  activation='relu', padding='same', name='decoder_h7')(decoder_h)
        decoder_h=BatchNormalization(name='decoder_h8')(decoder_h)
        decoder_h=UpSampling2D(name='decoder_h9')(decoder_h)
        x_hat=Conv2D(1 , kernel_size=(3,3), activation='sigmoid', padding='same', name='decoder_h10')(decoder_h)
        decoder=Model(z, x_hat)
        decoder.summary()
        return decoder
    
    def train(self):
        #data load, 데이터 불러오기
        (x_train, _ ), (x_test, y_test ) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        x_train = x_train.reshape(-1,*self.input_shape)
        x_test  = x_test.reshape(-1,*self.input_shape)
        
        self.dcvae.fit(x_train, x_train,shuffle=True, epochs=self.epochs, batch_size=self.batch_size, validation_data=(x_test, x_test))        
        self.dcvae.save_weights('./vae_model.hdf5')        
                        

        
    def visualization(self):    
        self.dcvae.load_weights('c:/data/vae/vae_model.hdf5')
        (x_train, _ ), (x_test, y_test ) = mnist.load_data()
        
        ##visualization
        x_test_encoded = self.encoder.predict(x_test, batch_size=self.batch_size)
        #plt.style.use('ggplot')
        plt.style.use('classic')
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
        plt.colorbar()
        plt.show()
        plt.close()
    

        ##visualization
        n = 30  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # we will sample n points within [-15, 15] standard deviations
        grid_x = np.linspace(-2, 2, n)
        grid_y = np.linspace(-2, 2, n)
        
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]]) * self.epsilon_stddev
                x_decoded = self.decoder.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
        plt.style.use('grayscale')
        plt.figure(figsize=(10, 10))
        plt.imshow(figure)
        plt.show()
        plt.close()

        
        
a=VAE()
a.train()


import os
os.getcwd()



 