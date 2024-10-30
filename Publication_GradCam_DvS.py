'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'### The following script is made as part of the study #Video-based robotic surgical action recognition and skills assessment on porcine models using deep learning###'
'### The code is open-source. However, when using the code, please make a reference to our paper and repository.""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'

###IMPORTS###
'Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes a heatmap' \
'explaining where the most contributing filter weights "looks" at the input image'
import random

import cv2
import keras_preprocessing.image

import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow.python.keras
from tensorflow import GradientTape
from tensorflow.python.keras.models import load_model
from numpy import load
import random
from numpy.random import randint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from PIL import Image
'### VARIABLES ###'
string_list = ['Suturing','Dissection']

img_path = r"PATH"

Y_testImg = np.load('TestImgLSTM_AR.npy', allow_pickle=True)
Y_testLabel = np.load('TestLabelLSTM_AR.npy', allow_pickle=True)

SIZE = 256

'### FUNCTIONS ###'
# function for generating samples
def load_testData(img_name, label, img_path):
    #label = np.array(label)
    #img_name = np.array(img_name)

    h = 1
    imgArray = []
    imgArray_LSTM = []
    label_LSTM = []
    for j in range(len(img_name)):
        for i in range(len(img_name[1-1])):
            FullName = img_path + img_name[j,i]
            print("Loading Image:", FullName, "image number", h, "of", len(img_name)*len(img_name[1-1]))
            img = cv2.imread(FullName[0], 1)
            img = Image.fromarray(img.astype(np.uint8))
            loadedImgSize = np.size(img)
            # Crop images to remove top an bottom information boxes (we want to learn from image data and not letters etc. in the info boxes (Unreliable))
            if loadedImgSize[1] == 1080:  # the img is 1080x1920
                img = img.crop((0, 165, 1920, 900))
            else:  # If not 1080x1920 then do this (assumed they are otherwise 1280x720) Might need universal fix here instead of hardcode defining image resolution.
                img = img.crop((0, 108, 1280, 612))
            img = np.array(img)
            img = cv2.resize(img, (SIZE, SIZE))
            img = img / 256
            imgArray.append(img)
            h =h+1
        imgArray_LSTM.append(imgArray)
        imgArray = []
        label_LSTM.append(label[j,0])
    imgArray_LSTM = np.array(imgArray_LSTM)
    label_LSTM = np.array(label_LSTM)
    return imgArray_LSTM, label_LSTM

'### STEPS ###'

def gradCAM(model, img, label, n_samples):
    model.summary()
    # choose random instances
    ix = randint(0, img.shape[0], 1)
    # retrieve selected images
    img, label = img[ix], label[ix]
    # 1. step : Define a model that has the last conv layer as input and last dense layer as output
    # here we crete two models with the same input and two outputs: last_conv_layer, original output --> softmax prediction
    grad_model = Model(
        [model.inputs], [model.get_layer('time_distributed_7').output, model.output]
    )
    # 2. step: Get the gradient of the class score (prediction) with respect to the weights of the last conv layer
    # given the input image

    heatmap_arr = []
    pred_label_arr = []
    true_label_arr = []
    for i in range(n_samples):
        img_single_patch = img
        img_single_patch = np.expand_dims(img_single_patch, axis=0)
        with GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img)
            pred_index = np.argmax(preds[0])
            class_channel = preds[:, pred_index]

        pred_label = string_list[pred_index]
        true_label = string_list[np.argmax(label)]

        gradients = tape.gradient(class_channel, last_conv_layer_output) # gradients are obtained here
        gradients1 = gradients[:,i:i+1,:,:,:]
        gradients1 = tf.convert_to_tensor(np.squeeze(gradients1, axis=0))
        pooled_gradients = K.mean(gradients1, axis=(0, 1, 2)) # they are avg pooled here

        last_conv_layer_output = last_conv_layer_output[0][i]
        heatmap = last_conv_layer_output @ pooled_gradients[..., tensorflow.newaxis]
        # In this ^ code, heatmap is assigned the result of a dot product between last_conv_layer_output and pooled_gradients
        # with the addition of a new axis. @ is the matrix multiplication operator in Python, and @ can also be written as
        # numpy.matmul if numpy is imported. last_conv_layer_output and pooled_gradients are two arrays. The dot product
        # of these two arrays is a scalar value, but with the use of the ... and tensorflow.newaxis syntax, the resulting
        # scalar is broadcasted to a new tensor with additional dimensions. ... is used to refer to all dimensions of the
        # original tensor and tensorflow.newaxis is used to add a new dimension to the tensor. This allows the scalar result
        # of the dot product to be broadcasted to a tensor of higher dimension, which can be used for other operations.
        heatmap = np.squeeze(heatmap) # remove the batch dimension
        heatmap = tensorflow.maximum(heatmap, 0) / tensorflow.math.reduce_max(heatmap) # normalize heatmap
        # rescale
        heatmap = np.uint8(255 * heatmap)

        # Colorize the colormap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Resize image such that it fits with the original image
        jet_heatmap = keras_preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((256, 256))
        jet_heatmap = np.array(jet_heatmap)
        heatmap_arr.append(jet_heatmap)
        pred_label_arr.append(pred_label)
        true_label_arr.append(true_label)
    heatmap_arr = np.array(heatmap_arr)
    pred_label_arr = np.array(pred_label_arr)
    true_label_arr = np.array(true_label_arr)
    return heatmap_arr, img, pred_label_arr, true_label_arr


img, label = load_testData(Y_testImg, Y_testLabel, img_path)

# 1. step : Load a pre-trained model and obtain the last conv layer and the final dense layer
# load model
modelName = 'Best_model_AR.h5'
model = keras.models.load_model(modelName)


# n_samples =5
# heatmap_arr, img_arr, pred_label_arr, true_label_arr = gradCAM(model, img=img, label=label, n_samples=n_samples)
#
# for i in range(n_samples):
#     plt.subplot(2, n_samples, 1 + i)
#     plt.axis('off')
#     plt.imshow(img_arr[0,i])
#     plt.imshow(heatmap_arr[i], alpha=0.4)
#     plt.title('Predicted label: ' + '\n' + pred_label_arr[i], fontsize=6)
# for i in range(n_samples):
#     plt.subplot(2, n_samples, 1 + n_samples + i)
#     plt.axis('off')
#     plt.imshow(img_arr[0,i])
#     plt.title('True label: ' + true_label_arr[i], fontsize=6)
# plt.tight_layout()
# plt.show()

while True:
    n_samples = 5
    heatmap_arr, img_arr, pred_label_arr, true_label_arr = gradCAM(model, img=img, label=label, n_samples=n_samples)

    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + i)
        plt.axis('off')
        plt.imshow(img_arr[0, i])
        plt.imshow(heatmap_arr[i], alpha=0.4)
        plt.title('Predicted label: ' + '\n' + pred_label_arr[i], fontsize=6)

    for i in range(n_samples):
        plt.subplot(2, n_samples, 1 + n_samples + i)
        plt.axis('off')
        plt.imshow(img_arr[0, i])
        plt.title('True label: ' + true_label_arr[i], fontsize=6)

    plt.tight_layout()
    plt.show()

    user_input = input("Press Enter to continue or type 'exit' to quit: ")
    if user_input.lower() == 'exit':
        break

print('Test completed')

