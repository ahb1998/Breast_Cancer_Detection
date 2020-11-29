"""
Created on Wed Nov 11 10:45:10 2020
# -*- coding: utf-8 -*-
@author: AHB_1998
@Project Name:  Breast Cancer Pathology with DeepLearning (PYHTON)
"""


# 1. Import Required Modules

import os
import glob
import keras
import random
import numpy as np
import tensorflow as tf
from keras.layers import *
import keras.backend as k
from keras.models import *
from keras.optimizers import *
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread, imshow, imsave
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping



# 2. Define Train & Test Path (Images + Mask Path for Train and Test Stages)

TRAIN_IMAGE_PATH = 'C:/Users/AHB_1998/Desktop/uni/T9/PRJCT/py-code/Dataset/Inputs_Train/'
TRAIN_MASK_PATH = 'C:/Users/AHB_1998/Desktop/uni/T9/PRJCT/py-code/Dataset/Masks_Train/'

TEST_IMGAE_PATH = 'C:/Users/AHB_1998/Desktop/uni/T9/PRJCT/py-code/Dataset/Inputs_Test/'
TEST_MASK_PATH = 'C:/Users/AHB_1998/Desktop/uni/T9/PRJCT/py-code/Dataset/Masks_Test/'


# 3. Initialize Images and Mask Size

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 64,64, 3



# 4. Define Pre_Processing Function (Region of Interest Extraction _ ROI)
#list ax ha:   
Train_Mask_List = sorted(next(os.walk(TRAIN_MASK_PATH))[2])
Test_Mask_List = sorted(next(os.walk(TEST_MASK_PATH))[2])

def Data_Proprocessing_Train():
    #ax asli:
    Init_Image = np.zeros((len(Train_Mask_List), 768, 896, 3), dtype = np.uint8)
    #mask asli:
    Init_Mask = np.zeros((len(Train_Mask_List), 768, 896), dtype = np.bool)
    #ax crop shode:(patch asli)
    Train_X = np.zeros((len(Train_Mask_List), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    #mask crop shode:(patch mask)
    Train_Y = np.zeros((len(Train_Mask_List), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
    
    n = 0
    #name all mask haye vaqe dar path ro mirizim dakhel var:
    for mask_path in glob.glob('{}/*.TIF'.format(TRAIN_MASK_PATH)):
        
        #extract name ax (os.basename>>tail mide(name), os.dirname>>ta qbl un esm ru mide)
        base = os.path.basename(mask_path)
        #2ta var mizarim, os.split miad name file ro az format juda mikone
        image_ID, ext = os.path.splitext(base)
        #miam img , mask motanazer ro negasht mikonm.
        image_path = '{}/{}_ccd.tif'.format(TRAIN_IMAGE_PATH, image_ID)
        #in ax ru mikhune va mirize tu mask
        mask = imread(mask_path)
        #in ax ru mikhune va mirize tu image
        image = imread(image_path)
        
        
        #hala bayad location pixel haye white dar mask taien bshe(patch)
        #np.where miad mokhtasaate ye pixel ba vizhegi k ma besh migim ru peida mikone:
        y_coord, x_coord = np.where(mask == 255)
        
        y_min = min(y_coord) 
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)
        
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        #inja miam taqirsize ru emal mikonim.
        Train_X[n] = resize(cropped_image[:,:,:IMG_CHANNELS],
               (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
               mode = 'constant',
               anti_aliasing=True,
               preserve_range=True)
        #arguman : "1":dune dune begir "2":output>size pix tarifi 
        #"3": preserve_range : agar True bashe ,pix resize shude meqdar hash ru hefz mikone
        #agar False bashe momayezi mikone(yani az -1 ta 1 mishan)
        #ye option dg ham hast be name "anti_aliasing" : vaqti pix ru down scale mikoni , baraye inke labe 
        #haye tasvir saaf tar benzr brsn --meqdaresh mitone True/False bashe.
      

        Train_Y[n] = np.expand_dims(resize(cropped_mask, 
               (IMG_HEIGHT, IMG_WIDTH),
               mode = 'constant',
               anti_aliasing=True,
               preserve_range=True), axis = -1)
        #arguman : "1":dune dune begir "2":output>size pix tarifi 
        #"3": preserve_range : agar True bashe ,pix resize shude meqdar hash ru hefz mikone
        #agar False bashe momayezi mikone(yani az -1 ta 1 mishan)
        #ye option dg ham hast be name "anti_aliasing" : vaqti pix ru down scale mikoni , baraye inke labe 
        #haye tasvir saaf tar benzr brsn --meqdaresh mitone True/False bashe.
       
        #hala miam in haru dar yek Tensor save mikonim 
        Init_Image[n] = image
        Init_Mask[n] = mask
        
        n+=1
        
    return Train_X, Train_Y, Init_Image, Init_Mask

Train_Inputs, Train_Masks, Init_Image, Init_Mask = Data_Proprocessing_Train()


#in hamun function balast ke baraye tasaavir test inkaro mikonim: 
def Data_Proprocessing_Test():
    
   
    Test_X = np.zeros((len(Test_Mask_List), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
    Test_Y = np.zeros((len(Test_Mask_List), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
    
    n = 0
    
    for mask_path in glob.glob('{}/*.TIF'.format(TEST_MASK_PATH)):
        
        base = os.path.basename(mask_path)
        image_ID, ext = os.path.splitext(base)
        image_path = '{}/{}_ccd.tif'.format(TEST_IMGAE_PATH, image_ID)
        mask = imread(mask_path)
        image = imread(image_path)
        
        y_coord, x_coord = np.where(mask == 255)
        
        y_min = min(y_coord) 
        y_max = max(y_coord)
        x_min = min(x_coord)
        x_max = max(x_coord)
        
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        
        Test_X[n] = resize(cropped_image[:,:,:IMG_CHANNELS],
               (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
               mode = 'constant',
               anti_aliasing=True,
               preserve_range=True)
        
        Test_Y[n] = np.expand_dims(resize(cropped_mask, 
               (IMG_HEIGHT, IMG_WIDTH),
               mode = 'constant',
               anti_aliasing=True,
               preserve_range=True), axis = -1)
        
        
        n+=1
        
    return Test_X, Test_Y

Test_Inputs, Test_Masks = Data_Proprocessing_Test()

        
    # 4.1. Show The Results in Preprocessing Stage
    
# print('Original_Image')
# imshow(Init_Image[0])
# plt.show()

# print('Original_Mask')
# imshow(Init_Mask[0])
# plt.show()

# print('Region_of_Interest_Image')
# imshow(Train_Inputs[0])
# plt.show()

# print('Region_of_Interest_Mask')
# imshow(np.squeeze(Train_Masks[0]))
# plt.show()

rows = 1
columns = 4
Figure = plt.figure(figsize=(15,15))
Image_List = [Init_Image[0], Init_Mask[0], Train_Inputs[0], Train_Masks[0]]

for i in range(1, rows*columns + 1):
    Image = Image_List[i-1]
    Sub_Plot_Image = Figure.add_subplot(rows, columns, i)
    Sub_Plot_Image.imshow(np.squeeze(Image))
plt.show()


# 5. Implementation of U_NET Model for Semantic Segmentation

def U_Net_Segmentation(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    #use keras core for making keras compatible input tensor
    inputs = Input(input_size)
    #bayad tensor keras ru vared layer haye Unet konim
    #normalize data

    n = Lambda(lambda x:x/255)(inputs)
    #baraye model sazi bekhatere zaf ee skht afzar feature haru bax tasvir
    # (64-128-256-1024)  be (16-32-64-128) taqir midm.
    """
    c: Convolution2D(filterSize(3*3)-activation(elu(exponential-Linear-Unit)farq esh ba relu ine k 
    jahat girish nesbat b manfi ,smooth tar)
    -raftare RELU nesbat b Negetive ha Sharp tare-
    kernel initializer >>be manzur vazn dehi avalie be kernel ha hast."he_normal"miad ba estefade az
    ye tabe normal anjam mide.
    pading>> valid ya same : valid bezarim yani padding nadarim , agar same bezarim yani 
    feature map khoruji size esh ba feature map vorodi yeki bashe
    stride ham mishe gaam e harekat (default 1 ee) maam 1 mikhaym pas nminevisim.
    
    p: Pooling
    
    
    u: upSampeling
    
    """

    
    c1 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(n)
   
    """in n ru mizarim ta begim ru kodom layer in karo
    anjam bede
    ye ravesh saade bara jologiri az OVERFITING dar phase training estefade az Dropout ee
    Ravesh kar : Tamaame featureMap haru mojadad dar convu badi  Bazi nmide
   va ba ye nesbati unaro door mirize            
    """
    c1 = Dropout(0.1)(c1)# in c1 ke minevisim manzur ine k darim migim ru in layer anajm bde
    c1 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    #block aval chun dota convolution dashtim hamun code bala ru copy/paste kardim
    # tanha farqi k krd miaim in taqirat ru bejaye "n" ruye "c1"anjam midim (*)     



    c2 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(p1)
    #chera p1? chun ru marhale qabl karemon ro mibarim jolo
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)


    c3 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    #inja 0.2 gozashtim chun tedaade filter ha ziad tar shd
    c3 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)


    c4 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)


    c5 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c5)


    """      
   inja kare encoding tamum mishe
   az inja be baad mirim vase upSampeling ya decoding
   """

    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    #niaz b concatination block shomare 6 va shomare 4:
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c6)   


    u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c7) 

    u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c8) 
    
    
    u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis = 3)
    c9 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3,3), activation='elu', kernel_initializer='he_normal',
                padding='same')(c9) 
    
    outputs = Conv2D(1,(1,1), activation='sigmoid')(c9)
    #chera darim az sigmoid estefade mikonim ?:
    #chun masale 2 classe ast,,,, mishud az softmax estefade krd
    #vali softmax vaswe chnd classe ast va niazi ndrim 
    
    #inja miaim model ru misazim 

    model = Model(inputs=[inputs], outputs=[outputs])
   #frayand train ro shru mikonim
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=[Mean_IOU_Evaluator])
   #optimize : 
   #metrics : metric yi hast k khudemon misazim (khat)
    model.summary()
    return model
    
    
# 6. Define U_NET Model Evaluator (Intersection Over Union _ IOU)

def Mean_IOU_Evaluator(y_true, y_pred):
    #precision -deqat
    prec = []
        
    #t be onvan shomarande (treshold)
    for t in np.arange(0.5, 1, 0.05):
        
        y_pred_ = tf.to_int32(y_pred>t)
        #meqdar predict ro ba treshold compare
        #chun in meqdar float ee miam ba tf.to_int32 tabdil b int mikonm
        
        
        #miaim dar har epoch  metric IOU ro mohasebe mikone.
        #output : 2ta out mide :1-tensori k avg IOU 2-UPDATE OPREATION 
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        #ijad 1 tensorflow sesion va define local var k mortabet ba TensorFlow
        # k inja keras ee
        k.get_session().run(tf.local_variables_initializer())
       
        #ba tavajoh b control mohtavayi k qarare montasher bshe ba matrix confusion
        #bayad malom konim in matrix ba opration balayi bas kar kone 
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return k.mean(k.stack(prec), axis = 0)
        #ba estefade az k.stack() una ro save krdim
        #dar akhar miangin anasor in tensor dar boad aval un
        #ro return krdim

model = U_Net_Segmentation()

            
# 7. Show The Results per Epoch


class loss_history(keras.callbacks.Callback):
#ba keras.callbacks.callback, 6 option dar ekhtiar migiram.
#1-onTrainBegin,2-onTrainEnd, 3-onEpochBegin
#4-onEpochEnd, 5-onBatchBegin,6-onBatchEnd
  
    def __init__ (self, x=4):    #mikaym begim ravand har epoch ru neshun bede
        self.x = x
     
    #chun mikhaym tu har epoch result ru bebinim, az shomare3 estefade mikonim   
    def on_epoch_begin(self, epoch, logs={}):
        
        #imshow(Train_Inputs[self.x])
        #plt.show()
        
        #imshow(np.squeeze(Train_Masks[self.x]))
        #plt.show()
        
        preds_train = self.model.predict(np.expand_dims(Train_Inputs[self.x], axis = 0))
        #imshow(np.squeeze(preds_train[0]))
        #plt.show()
  

imageset = 'BCC'
backbone = 'UNET'
version = 'v1.0'
model_h5 = 'model-{imageset}-{backbone}-{version}.h5'.format(imageset=imageset, 
                  backbone = backbone, version = version)
model_h5_checkpoint = '{model_h5}.checkpoint'.format(model_h5=model_h5)

earlystopper = EarlyStopping(patience=7, verbose=1)
"""
early stopper: payane farayand amoozesh amuzesh darsurati k khata behbudi peida nkrd
earlystopper = EarlyStopping(patience=7, verbose=1)
patience:ye digit k taiin mikonim ,harvaqt behbudi hasel nshd faraynd amoozesh motevaqef
verbose: 2 mode dre(0:vqti shoma ravand kar ru dr amozesh nmikhay bbini)(1:ravand amoozesh qabel didane)
checkpoint ham baraye save model dar surat behbud model 

"""
checkpointer = ModelCheckpoint(model_h5_checkpoint, verbose = 1, save_best_only=True)

  
    
# 8. Train U_NET Model using Training Samples

#result amoozesh ru tuye in var mirizm
results = model.fit(Train_Inputs, Train_Masks, 
                    validation_split=0.1, 
                    batch_size=2,
                    epochs=50,
                    callbacks=[earlystopper, checkpointer, loss_history()])
#che tedad az sample amoozeshi bere bra validation ya arzyabi model 
#bemanzur jologiri az overfiting 
#batch_size=2,#pardazesh daste yi (baste be sakhafzar in adad mishe bala v payin bord)
#epochs=50,#tedad tekrar frayand amoozesh
    
# 9. U_NET Model Evaluation using Test Samples

#arzyabi model Unet , ham bara amoozeshi , ham bara azmayeshi
preds_train = model.predict(Train_Inputs, verbose=1)
preds_train_t = (preds_train>0.5).astype(np.uint8)
preds_test = model.predict(Test_Inputs, verbose=1)
preds_test_t = (preds_test>0.5).astype(np.uint8)
    
# 10. Show Final Results (Segmented Images)
#namayesh tasadofi yek mored amoozesh 
#ye adad random
ix = random.randint(0, len(Train_Inputs)-1)

print(ix)

print('Train_Image')
imshow(Train_Inputs[ix])
plt.show()

print('Train_Mask')
imshow(np.squeeze(Train_Masks[ix]))
plt.show()

print('Segmented_Image')
imshow(np.squeeze(preds_train[ix]))
plt.show()


iix = random.randint(0,1)
print(iix)

print('Test_Image')
imshow(Test_Inputs[iix])
plt.show()

print('Test_Mask')
imshow(np.squeeze(Test_Masks[iix]))
plt.show()

print('Segmented_Test_Mask')
imshow(np.squeeze(preds_test[iix]))
plt.show()

"""
elat zaiif budn output:
    1-tedad payin input bara train (50)
    2-kahesh size tasvir bara afzayesh sorat fit shdn model
    3-model U-Net ro ba 16 filter shru krdim
    
"""

# 11. Show Loss and IOU Plots


# 11.1. Summarize History for Loss

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['Training','Validation'], loc = 'upper left')
plt.show()


# 11.1. Summarize History for IOU

plt.plot(results.history['Mean_IOU_Evaluator'])
plt.plot(results.history['val_Mean_IOU_Evaluator'])
plt.title('Intersection Over Union')
plt.ylabel('IOU')
plt.xlabel('epochs')
plt.legend(['Training','Validation'], loc = 'upper left')
plt.show()
























