## About

This code is implementation of the paper https://arxiv.org/pdf/1703.05165.pdf on ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection, Part 1: Lesion Segmentation.

Competition link: https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a
---

## Data set folders

1. **trainx** - 2000 training RGB images of effected parts of skin resized to 192 x 256.
2. **trainx** - 2000 segmented (Ground Truth) images of images in train resized to 192 x 256.
3. **validationx** - 150 validation images of effected parts of skin resized to 192 x 256.
4. **validationy** - 150 segmented (Ground Truth) images of images in train resized to 192 x 256.
5. **testx** - 600 test images of effected parts of skin resized to 192 x 256.
6. **testy** - 600 segmented (Ground Truth) images of images in train resized to 192 x 256.

All the images in all the folders are resized to 192 x 256 size. 

If you want to resize the original images in data set then use the **reshape.py** script by doing minute changes in it like changing folder names and number of images.

Here you dont need to run the **reshape.py** on images as the images in the folders are already resized to our required dimensions i.e, 192 x 256 (Can see the reason for this in the original paper)

---

## Code

1. **reshape.py** - code to reshape all the images in a folder to 192 x256 images.(No need to run in this case)
2. **melanoma_segmentation.ipynb** - Jupyter notebook with all the code for convolution and deconvolution layers, training, etc.
3. **melanoma_segmentation.py** - Python script with same code as in **melanoma_segmentation.ipynb** jupyter notebook.
4. **load_batches.py** - load all the images in a folder to batchs ao batch size 16.(No need to run this script as another function which do the same operation as this is created in the jupyter notebook)

---

## Errors

1. (**Solved**)An error with the axis in the tf.reduce-sum() operation in the **jaccard_loss()** function which I created.

2. (**Solved**)There is some problem with dividing the input images into batches. Because of which a dimentional error is occuring.

3. (**Solved**)There is a problem with the function **unpool()** which need to be fixed.

   Created new function called **UnPooling2x2ZeroFilled()** instead of **unpool()** for unpooling layer during deconvolution which is working fine.

---

## Note

The code after the line **_, _, parameters = model(X_train, Y_train, X_test, Y_test)** after the **model()** function in the main jupyter notebook is just spare code and some testing so **don't** run the code after the above mentioned line of code
