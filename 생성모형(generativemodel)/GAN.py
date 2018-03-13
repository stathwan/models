"""
  GAN(Generative Adversarial Nets)은 관심 있는 데이터의 분포를 모사하는 생성모형(Generative model)의 
  일종이다. GAN은 생성모형 G(Generator model; 다층신경망)과 판별모형 D(Discriminative model; 다층신경망)으로 
  구성되어있으며 대립적(adversarial)인 절차(학습)을 통해 원하는 데이터를 생성한다. 
  본 코드에서는  GAN이 제안된 논문 Ian J. Goodfellow, 2014, Generative Adversarial Nets,arXiv',
  git https://github.com/eriklindernoren/Keras-GAN,
  코딩셰프의 3분 딥러닝을 참고하여   
  한국연애인 이미지를 생성하는 GAN모형을 만들었다.
"""

#모듈 load
import os
import glob
import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input, BatchNormalization, Reshape, Dropout
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt


class FaceGAN:
    def __init__(self,chennel=1):
        self.pyroot = os.getcwd()
        self.impath=glob.glob(self.pyroot+'\\data\\*')    
        
        self.img_size=64
        self.chennel=chennel
        self.image_shape = (self.img_size,self.img_size,self.chennel)
        self.noise_shape = (100,)

        self.face_img = self.load_data()        
        self.Generator=self.Generator_model()
        self.Discriminator=self.Discriminator_model()
        self.Generator_Discriminator=self.Generator_Discriminator_model()
        
    def load_data(self):
        face_img=np.empty(shape=[1,self.img_size,self.img_size,3])
        
        for i in range(len(self.impath)) :
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
        
    #생성망
    def Generator_model(self):
        model= Sequential()
        model.add(Dense(256, input_shape=self.noise_shape))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dense(512))
        model.add(LeakyReLU())     
        model.add(BatchNormalization())
        model.add(Dense(1024))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dense(np.prod(self.image_shape), activation= 'tanh'))
        model.add(Reshape(self.image_shape))
        model.compile(loss= 'binary_crossentropy' , optimizer = 'Adam', metrics=['accuracy'])
        return(model)
        
    #판별망
    def Discriminator_model(self):
        model= Sequential()
        model.add(Flatten(input_shape=self.image_shape))
        model.add(Dense(512))
        model.add(LeakyReLU())
        model.add(Dropout(rate=0.5))
        model.add(Dense(256))
        model.add(LeakyReLU())
        model.add(Dropout(rate=0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss= 'binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return(model)
        
    #생성망-판별망
    def Generator_Discriminator_model(self):

        self.Discriminator.trainable=False
        
        latent_z=Input(shape=self.noise_shape)
        generated_image=self.Generator(latent_z)
        value=self.Discriminator(generated_image)
        
        Generator_Discriminator= Model(latent_z, value)
        Generator_Discriminator.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return(Generator_Discriminator)
        
    #학습
    '''
    GAN모형은 판별망과 생성망의 학습이 따로 진행된다. 먼저 판별망의 학습을 진행하고 그 후 생성망의 학습을 진행한다. 
    '''
    def train(self,epochs=10000,batch_size=100,save_interval=100):


        
        half_batch= int(batch_size/2)
        
        D_loss_log= [] # loss를 저장하기 위한 리스트
        G_loss_log=[] 
        epoch=0
        ##판별망 학습하기
        for epoch in range(epochs):
            #판별함수의 가중치를 업데이트 가능하게 한다.
        
            #훈련 이미지 추출 
            index = np.random.randint(0, self.face_img.shape[0], half_batch)
            real_image= self.face_img[index]
            
            # 가짜 이미지 생성
            noise = np.random.normal(0,1,(half_batch,100))
            fake_image= self.Generator.predict(noise)
            
            # 가짜이미지와 훈련이미지 합치기
            labels_rf= np.array([[0]*half_batch,[1]*half_batch]).reshape(-1,)
            ImgForTrain=np.concatenate((fake_image, real_image))
            Discriminator_loss = self.Discriminator.train_on_batch(ImgForTrain, labels_rf)
            
            for i in range(10):
                ##생성망 학습하기
                noise = np.random.normal(0,1,(batch_size, 100))
                labels_fake= np.ones((batch_size,), dtype='int') # 학습을 위해 생성된 결과의 레이블은 1로 한다.
                Generator_loss=self.Generator_Discriminator.train_on_batch(noise, labels_fake)[0]
                
            ##학습결과
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %(epoch, Discriminator_loss[0], 100*Discriminator_loss[1], Generator_loss))
            D_loss_log.append([epoch, Discriminator_loss])
            G_loss_log.append([epoch, Generator_loss])
            
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
            
    #생성
    def save_imgs(self,epoch):
        img_size=self.img_size
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.Generator.predict(noise)
        
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
        fig.savefig(self.pyroot+ "/result/face_%d.png" % epoch)
        plt.close()
        

def main():
    m=FaceGAN()
    m.train()
    
if __name__ == '__main__' :
    main()