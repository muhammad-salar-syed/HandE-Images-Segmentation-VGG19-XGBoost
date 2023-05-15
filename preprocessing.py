'''
Original MATLAB code:
    https://github.com/mitkovetta/staining-normalization/blob/master/normalizeStaining.m


http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf    
Input: RGB image
Step 1: Convert RGB to OD
Step 2: Remove data with OD intensity less than β
Step 3: Calculate  singular value decomposition (SVD) on the OD tuples
Step 4: Create plane from the SVD directions corresponding to the
two largest singular values
Step 5: Project data onto the plane, and normalize to unit length
Step 6: Calculate angle of each point wrt the first SVD direction
Step 7: Find robust extremes (αth and (100−α)th 7 percentiles) of the
angle
Step 8: Convert extreme values back to OD space

'''

import glob
import os
from tifffile import tifffile
from skimage.transform import resize
from skimage import io,img_as_float,img_as_int
from skimage.filters import threshold_otsu
import numpy as np
import cv2
import matplotlib.pyplot as plt

Io = 240 # Normalizing factor for image intensities
alpha = 1  # Tolerance for the pseudo-min and pseudo-max (default: 1)
beta = 0.15 # OD threshold for transparent pixels (default: 0.15)

HERef = np.array([[0.5626, 0.2159],  # H&E OD matrix
                  [0.7201, 0.8012],
                  [0.4062, 0.5581]])

maxCRef = np.array([1.9705, 1.0308])  # Maximum stain concentrations for H&E


img_path=[]
mask_path=[]
for root,dirs,files in os.walk('.'):
    for i in dirs:
        if i == 'mask binary without border':
            x=[glob.glob(str(os.path.join(root,str(i))) + '/*')]
            for a in x:
                mask_path.append(a)
            
        if i == 'tissue images':
            y=[glob.glob(str(os.path.join(root,str(i))) + '/*')]
            for b in y:
                img_path.append(b)
                
Images=[]         
for path in img_path:
    for i in path:
        img=cv2.imread(i, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r_img=cv2.resize(img, (384,256)) 
        h, w, c = r_img.shape
        I = r_img.reshape((-1,3))
        OD = -np.log10((I.astype(np.float)+1)/Io)
        ODhat = OD[~np.any(OD < beta, axis=1)]  # remove transparent pixels (clear region with no tissue)
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))  # compute eigen values & eigenvectors
        That = ODhat.dot(eigvecs[:,1:3])  # Create plane from the SVD directions
        phi = np.arctan2(That[:,1],That[:,0])
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100-alpha)
        vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
        if vMin[0] > vMax[0]:    
            HE = np.array((vMin[:,0], vMax[:,0])).T
        else:
            HE = np.array((vMax[:,0], vMin[:,0])).T
        
        Y = np.reshape(OD, (-1, 3)).T
        C = np.linalg.lstsq(HE,Y, rcond=None)[0]
        maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
        tmp = np.divide(maxC,maxCRef)
        C2 = np.divide(C,tmp[:, np.newaxis])
        Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
        Inorm[Inorm>255] = 254
        Inorm = np.reshape(Inorm.T, (h,w,3)).astype(np.uint8)  
        H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
        H[H>255] = 254
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
        E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
        E[E>255] = 254
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
        
        Images.append(Inorm)


Masks=[]
for path in mask_path:
    for i in path:
        mask=cv2.imread(i,0)
        r_img=cv2.resize(mask, (384,256))
        Masks.append(r_img)
        
import random
num=random.randint(0,len(Images))
plt.subplot(121)
plt.imshow(Images[num])
plt.subplot(122)
plt.imshow(Masks[num],cmap='gray')


for i in range(len(Images)):
    tifffile.imwrite('./Images/'+'Image_{}.tif'.format(i), Images[i])
    tifffile.imwrite('./Masks/'+'Mask_{}.tif'.format(i), Masks[i])


