import os
import numpy as np
import glob
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.pyplot as plt
from PIL import Image
#from skimage import transform


filenamesx = [imgx for imgx in glob.glob("validationx/*.jpg")]

filenamesx.sort()

i = 1
imagesx = []
imagesx_resized = []
for imgx in filenamesx:
    if not imgx.endswith("superpixels.png" or ".txt"):
        imgsx = plt.imread(imgx)
        imgsx_resize = imresize(imgsx, (192, 256))
        #imgsx_resize = skimage.transform.resize(imgx, (192, 256))
        locals()["imgx"+str(i)] = imgsx_resize
        i = i+1
        print(imgx)
        
for i in range(150):
    imagesx_resized.append(locals()["imgx"+str(i+1)])
    
for i in range(150):
    imx = Image.fromarray(locals()["imgx"+str(i+1)])
    imx.save("validationx1/imgx"+str(i+1)+".jpg")
    
filenamesy = [imgy for imgy in glob.glob("validationy/*.jpg")]

filenamesy.sort()

j = 1
imagesy = []
imagesy_resized = []
for imgy in filenamesy:
    imgsy = plt.imread(imgy)
    imgsy_resize = imresize(imgsy, (192, 256))
    #imgsy_resize = skimage.transform.resize(imgy, (192, 256))
    locals()["imgy"+str(j)] = imgsy_resize
    j = j+1
    print(imgy)
        
for j in range(150):
    imagesy_resized.append(locals()["imgy"+str(j+1)])
    
for j in range(150):
    imy = Image.fromarray(locals()["imgy"+str(j+1)])
    imy.save("validationy1/imgy"+str(j+1)+".jpg")