from skimage import io,img_as_float,img_as_int
import glob
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.models import Sequential, Model

saved_img=glob.glob('./Images/*')
saved_mask=glob.glob('./Masks/*')

I=[]
M=[]
for i in range(len(saved_img)):
    img=io.imread(saved_img[i])/255.
    mask=io.imread(saved_mask[i],as_gray=True)/255.
    I.append(img)
    M.append(mask)
    
I=np.array(I)
M=np.array(M)
M=np.expand_dims(M, axis=3)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(I, M, test_size=0.1)

model = VGG19(weights='imagenet', include_top=False, input_shape=(512,512,3))
for layer in model.layers:
	layer.trainable = False
    
model.summary()

VGG19_model = Model(inputs=model.input, outputs=model.get_layer('block1_conv2').output)
VGG19_model.summary()

features = VGG19_model.predict(X_train)

X = features.reshape(-1, features.shape[3])
Y = Y_train.reshape(-1)

#XGBOOST
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X, Y) 

import pickle
pickle.dump(model, open('XG_model.sav', 'wb'))