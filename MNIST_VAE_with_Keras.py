# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:19:25 2018

@author: 2017B221
"""

# import module, 모듈 불러오기 
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Flatten, Reshape, UpSampling2D, Conv2DTranspose
from keras.models import Model 
from keras import backend as K 
from keras.engine.topology import Layer
from keras import metrics
from keras.datasets import mnist

# hyperparameter, 초모수설정
K.set_image_data_format('channels_first')
input_shape=(1,28,28)
latent_size=2 # z dim
epsilon_stddev= 1.0
batch_size=200
epochs=50

# 필요한 Layer 만들기 
class sampling(Layer):
    
    def __init__(self, latent_size=4,epsilon_stddev= 1.0, **kwargs):
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
        KLDivergence = 0.5*K.sum(1.+ log_variance_hat - K.square(mu_hat) - K.exp(log_variance_hat), axis=-1)
        loss = K.mean(-KLDivergence + ReconstructionError)
        
        return loss

    
#data load, 데이터 불러오기
(x_train, _ ), (x_test, y_test ) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(-1,*input_shape)
x_test  = x_test.reshape(-1,*input_shape)

#encoder, 인코더

x= Input(input_shape)
encoder_h= Conv2D(32 , kernel_size=(3,3), activation='relu', padding='same')(x)
encoder_h= Conv2D(64, kernel_size=(3,3), strides=(2,2), activation='relu', padding='same')(encoder_h)
encoder_h= Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',)(encoder_h)
encoder_h= Conv2D(64, kernel_size=(3,3), activation='relu', padding='same',)(encoder_h)
encoder_h= Flatten()(encoder_h)
encoder_h= Dense(32)(encoder_h)

mu_hat= Dense(latent_size,activation='linear')(encoder_h) # default latent_size=32

# VAE Loss function has log(variance) such that variance should be >0. To make it calculable We train log_variance  
log_variance_hat= Dense(latent_size, activation='linear')(encoder_h) 
# latent z, 잠재변수 z  
z=sampling(latent_size,epsilon_stddev)([mu_hat, log_variance_hat])

#decoder, 디코더

decoder_h=Dense(14*14*64, activation='relu', name='decoder_h1')(z) # to make generator
decoder_h=Reshape((64,14,14), input_shape=(14*14*64,), name='decoder_h2')(decoder_h)
decoder_h=Conv2DTranspose(32 , kernel_size=(3,3),  activation='relu', padding='same', strides=(2,2), name='decoder_h3')(decoder_h)
x_hat=Conv2D(1 , kernel_size=(3,3), activation='sigmoid', padding='same', name='decoder_h4')(decoder_h)
convae=Model(x, x_hat)

#compile 
vaeloss=VAELoss(input_shape,mu_hat,log_variance_hat).convloss
convae.compile(loss=vaeloss, optimizer='rmsprop')
convae.summary()
    
#fit & save model
convae.fit(x_train,x_train,shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))
convae.save_weights('c:/data/vae/vae_model.hdf5')

convae.load_weights('c:/data/vae/vae_model.hdf5')
#encoder
encoder = Model(x, z)
##visualization
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

#decoder
decoder_input = Input(shape=(latent_size,))
_x=convae.get_layer("decoder_h1")(decoder_input)
_x=convae.get_layer("decoder_h2")(_x)
_x=convae.get_layer("decoder_h3")(_x)
_x_hat=convae.get_layer("decoder_h4")(_x)
generator = Model(decoder_input, _x_hat)

##visualization
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# we will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]]) * epsilon_stddev
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()







