## About

This code is implementation of the paper "[Automatic skin lesion segmentation with fully convolutional-deconvolutional networks](https://arxiv.org/pdf/1703.05165.pdf)" on ISIC 2017: Skin Lesion Analysis Towards Melanoma Detection, Part 1: Lesion Segmentation.

This code is implemented using [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) frameworks.

## Data set folders

- [`trainx`](trainx) - 2000 training RGB images of effected parts of skin resized to 192 x 256.
- [`trainy`](trainy) - 2000 segmented (Ground Truth) images of images in train resized to 192 x 256.
- [`validationx`](validationx) - 150 validation images of effected parts of skin resized to 192 x 256.
- [`validationy`](validationy) - 150 segmented (Ground Truth) images of images in train resized to 192 x 256.
- [`testx`](testx) - 600 test images of effected parts of skin resized to 192 x 256.
- [`testy`](testy) - 600 segmented (Ground Truth) images of images in train resized to 192 x 256.

All the images in all the folders are resized to 192 x 256 size. 

If you want to resize the original images in data set then use the [`reshape.py`](reshape.py) script by doing minute changes in it like changing folder names and number of images.

Here you dont need to run the [`reshape.py`](reshape.py) on images as the images in the folders are already resized to our required dimensions i.e, 192 x 256 (Can see the reason for this in the original paper)

## Code

- [`reshape.py`](reshape.py) - code to reshape all the images in a folder to 192 x256 images.(No need to run in this case as the data in this repository is already reshaped)
- [`melanoma_segmentation.ipynb`](melanoma_segmentation.ipynb) - **Jupyter notebook with all the code for convolution and deconvolution layers, training, etc.**
- [`load_batches.py`](load_batches.py) - (Don't use this)load all the images in a folder to batchs of batch size 16.(No need to run this script as another function which do the same operation as this is created in the jupyter notebook)

## Errors

- (**Solved**)An error with the axis in the tf.reduce-sum() operation in the `jaccard_loss()` function which I created.

- (**Solved**)There is some problem with dividing the input images into batches. Because of which a dimentional error is occuring.

- (**Solved**)There is a problem with the function `unpool()` which need to be fixed.

   Created new function called `UnPooling2x2ZeroFilled()` instead of `unpool()` for unpooling layer during deconvolution which is working fine.

## Note

- The code after the line ``_, _, parameters = model(X_train, Y_train, X_test, Y_test)`` after the `model()` function in the main jupyter notebook is just spare code and some testing so **don't** run the code after the above mentioned line of code

- In [`melanoma_segmentation.ipynb`](melanoma_segmentation.ipynb), if you are running the jupyter notebook on your pc offline, **don't run** the cells with codes ``! git clone`` and ``cd``. Run there cells only if you are running this notebook in Google Colab.
