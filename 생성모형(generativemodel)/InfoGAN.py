"""
  본 코드에서는  논문 Xi Chen, InfoGan: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets,arXiv,2016',
  git https://github.com/eriklindernoren/Keras-GAN,
  youtube: PR-022: InfoGAN (OpenAI) 을 참고하여
  한국연애인 이미지를 생성하는 GAN모형을 만들었습니다.
"""

#모듈 load
import glob
import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, BatchNormalization, Reshape, Dropout, Conv2D,UpSampling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
import keras.backend as K

import matplotlib
matplotlib.use('Agg') # if threr are QXcbConnection: Could not connect to display pyplot error


import matplotlib.pyplot as plt
from keras.optimizers import Adam

class FaceDCGAN:
    def __init__(self):
        
        self.impath=glob.glob('./data2/*')
        
        self.chennel=3
        self.img_size=64
        self.image_shape = (self.img_size,self.img_size,self.chennel)

        self.nosie_latent_shape=(100,)
        self.num_classes = 2
        self.num_noise= int(self.nosie_latent_shape[0]-self.num_classes)
        self.noise_shape = (self.num_noise,)
        
        self.losses = ['binary_crossentropy', self.mutual_info_loss]
        self.optimizer = Adam(0.0002, 0.5)
        
        self.epochs=1000000
        self.batch_size=128
        self.save_interval=1000
            
        self.face_img = self.load_data()        
        self.Generator=self.Generator_model() # 생성모형 (판별모형 학습시 test 이미지 생성, 훈련 후 test이미지 생성)
        self.Discriminator, self.auxiliary= self.Discriminator_auxiliary_model() # 판별모형+보조항 
        self.Generator_Discriminator_Auxiliary=self.Generator_Discriminator_Auxiliary_model() #생성모형과 판별모형 결합(생성모형 학슴 위함)
        
    def load_data(self):
        
        face_img=np.empty(shape=[1,self.img_size,self.img_size,3])
        
        for i in range(int(len(self.impath))) :
            img=cv2.imread(self.impath[i])
            img=img.reshape(1,self.img_size,self.img_size,3)
            if i ==0 :
                face_img= img
            else:   
                face_img=np.concatenate((face_img,img))
                # print(str(i+1),"번째 이미지가 합쳐졌습니다.")  
                
        face_img.astype(np.float32)
        if self.chennel==1:
            face_img=np.dot(face_img[...,:3], [0.299, 0.587, 0.114])
            face_img=face_img.reshape(*face_img.shape,1)
        
        face_img=(face_img - 127.5)/127.5
        return(face_img)                    
    
    ##참고
    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))
        return conditional_entropy + entropy
    
    
    #생성망
    def Generator_model(self):
        nosie_latent_shape=self.nosie_latent_shape
        chennel= self.chennel
        
        model= Sequential()
        model.add(Dense(256*4*4, activation='relu' ,input_shape=(nosie_latent_shape)))
        model.add(Reshape((4, 4, 256)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128,activation='relu' ,kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64,activation='relu' ,kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(32,activation='relu' ,kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(chennel,activation='tanh', kernel_size=3, padding="same"))
        model.summary()
        return(model)
        
    #판별망+보조항
    def Discriminator_auxiliary_model(self):

        image_shape= (self.img_size,self.img_size,self.chennel)        
        _img = Input(shape=image_shape)
        
        # 판별망과 보조항 공유층
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=image_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.6))
        model.add(Conv2D(128, kernel_size=3, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.6))
        model.add(Conv2D(256, kernel_size=3, strides=2))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.6))
        model.add(Conv2D(512, kernel_size=3, strides=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.6))
        model.add(Flatten())
        
        
        _dim_reduction=model(_img)
        
        #discriminator
        _value=Dense(1, activation='sigmoid')(_dim_reduction)
        Discriminator_model=Model(_img,_value)
        Discriminator_model.compile(loss= ['binary_crossentropy'], optimizer=self.optimizer, metrics=['accuracy'])        
        
        #auxiliary
        _auxiliary = Dense(128, activation='relu')(_dim_reduction)
        _latent_c=Dense(self.num_classes, activation='softmax')(_auxiliary)       
        auxiliary_model= Model(_img,_latent_c)
        auxiliary_model.compile(loss= [self.mutual_info_loss], optimizer=self.optimizer, meteics=['accuracy'])
        
        return Discriminator_model, auxiliary_model
        
    #생성망-판별망
    def Generator_Discriminator_Auxiliary_model(self):
        self.Discriminator.trainable=False
        _input=Input(shape=self.nosie_latent_shape)
        _generated_image=self.Generator(_input)
        _value=self.Discriminator(_generated_image)
        _latent_c= self.auxiliary(_generated_image)
        
        Generator_Discriminator_Auxiliary_model= Model(_input, [_value, _latent_c] )
        Generator_Discriminator_Auxiliary_model.compile(loss=self.losses, optimizer=self.optimizer, metrics=['accuracy'])
        return(Generator_Discriminator_Auxiliary_model)
        
    #학습
    '''
    GAN모형은 판별망과 생성망의 학습이 따로 진행된다. 먼저 판별망의 학습을 진행하고 그 후 생성망의 학습을 진행한다. 
    '''
    def train(self):
       
        half_batch= int(self.batch_size/2)
        
        Discriminator_loss_log= [] # loss를 저장하기 위한 리스트
        Generator_loss_log=[] 
        epoch=0

        for epoch in range(self.epochs):
            #판별함수의 가중치를 업데이트 가능하게 한다.
            
            ##판별망 학습하기       
            #훈련 이미지 추출 
            index = np.random.randint(0, self.face_img.shape[0], half_batch)
            real_image= self.face_img[index]
            
            # 가짜 이미지 생성
            noise_z = np.random.normal(0,1,(half_batch,self.num_noise))
            latent_c= np.random.randint(0,self.num_classes,half_batch).reshape(-1,1)
            latent_c= to_categorical(latent_c, num_classes=self.num_classes)
            input_value=np.concatenate((noise_z, latent_c), axis=1)
                
            fake_image= self.Generator.predict(input_value)
            
            # 가짜이미지와 훈련이미지 합치기
            labels_rf= np.array([[0]*half_batch,[1]*half_batch]).reshape(-1,)
            ImgForTrain=np.concatenate((fake_image, real_image))
            Discriminator_loss = self.Discriminator.train_on_batch(ImgForTrain, labels_rf)
            

            ##생성망 학습하기
            noise_z = np.random.normal(0,1,(self.batch_size,self.num_noise))
            latent_c= np.random.randint(0,self.num_classes,self.batch_size).reshape(-1,1)
            latent_c= to_categorical(latent_c, num_classes=self.num_classes)
            input_value=np.concatenate((noise_z, latent_c), axis=1)
            
            labels_fake= np.ones((self.batch_size,), dtype='int') # 학습을 위해 생성된 결과의 레이블은 1로 한다.
            Generator_loss=self.Generator_Discriminator_Auxiliary.train_on_batch(input_value, [labels_fake,latent_c])[0]
                
            ##학습결과
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %(epoch, Discriminator_loss[0], 100*Discriminator_loss[1], Generator_loss))
            Discriminator_loss_log.append([epoch, Discriminator_loss])
            Generator_loss_log.append([epoch, Generator_loss])
            
            if epoch % self.save_interval == 0:
                self.save_imgs(epoch,[[1,0]])
                self.save_imgs(epoch,[[0,1]])
 
    #생성
    def save_imgs(self,epoch,input_c):
        img_size=self.img_size
        r, c = 10, 10

        noise_z = np.random.normal(0,1,(r * c,self.num_noise))
        latent_c= np.repeat(input_c,r * c, axis=0)
        input_value=np.concatenate((noise_z, latent_c), axis=1)

        gen_imgs = self.Generator.predict(input_value)
        
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c,figsize=(15,15))
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        
        cnt = 0
        for i in range(r):
            for j in range(c):
                img=gen_imgs[cnt,...]
                if self.chennel==3:
                    b, g, r = cv2.split(img) 
                    img = cv2.merge([r,g,b])
                else:
                    img=img.reshape(img_size,img_size)
                axs[i,j].imshow(img,cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig( "./result/face_{}_{}.png".format(input_c,epoch))
        plt.close()
        

def main():
    m=FaceDCGAN()
    m.train()
    
if __name__ == '__main__' :
    main()