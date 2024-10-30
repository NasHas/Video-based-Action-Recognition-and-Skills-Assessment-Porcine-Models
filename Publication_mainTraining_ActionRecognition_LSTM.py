'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'### The following script is made as part of the study #Video-based robotic surgical action recognition and skills assessment on porcine models using deep learning###'
'### The code is open-source. However, when using the code, please make a reference to our paper and repository.""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'


import time
import cv2
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.utils import plot_model
from tensorflow import keras
from matplotlib import pyplot
from numpy import load
from numpy.random import randint
from PIL import Image

#Network imports (from different files) - Made like this so its easy to select model to use
from Publication_cnn_lstm import cnn_lstm_net


'### HYPERPARAMETERS ###'
EPOCHS = 50 #Number of times the full dataset is processed in training (if not early stopping is triggered)
BATCH_SIZE = 8 #Batch size (sequences to process in a batch
n_classes = 2 #Number of classes in the last layer (# of classes to predict)
SIZE = 256 #Final image size to be processed by the network
earlyStoppingVar = 5 #Number of epochs that must pass without finding a better model and early stopping is activated
Seq_len = 5

#Image path to all individual frames
img_path = r"PATH"

#Loading of all data (names of frames and labels from .csv files - not actual images here)
# Import of dataset to dataframe (Training).
header_listTrain = ["Frame name", "Sutur", "Dissektion"]
dfTrain = pd.read_csv("Finally_train_set.csv", names=header_listTrain)
# To see first 5 rows of the dataset
dfTrain.head()
# To know the data types of the variables.
dfTrain.dtypes

# Import of dataset to dataframe (Validation).
header_listValidation = ["Frame name", "Sutur", "Dissektion"]
dfValidation = pd.read_csv("Finally_val_set.csv", names=header_listValidation)
# To see first 5 rows of the dataset
dfValidation.head()
# To know the data types of the variables.
dfValidation.dtypes

# Import of dataset to dataframe (Test).
header_listTest = ["Frame name", "Sutur", "Dissektion"]
dfTest = pd.read_csv("Finally_test_set.csv", names=header_listTest)
# To see first 5 rows of the dataset
dfTest.head()
# To know the data types of the variables.
dfTest.dtypes

#Convert data into two individual variables (pr. datasplit) by selecting columns
Y_trainLabel = np.array(dfTrain.drop(["Frame name"], axis=1))
Y_trainImg = np.array(dfTrain.drop(["Sutur", "Dissektion"], axis=1))

Y_valLabel = np.array(dfValidation.drop(["Frame name"], axis=1))
Y_valImg = np.array(dfValidation.drop(["Sutur", "Dissektion"], axis=1))

Y_testLabel = np.array(dfTest.drop(["Frame name"], axis=1))
Y_testImg = np.array(dfTest.drop(["Sutur", "Dissektion"], axis=1))

def LSTM_Prep(data, seq_len):
    num_sequences = len(data) // seq_len

    # list to store the sequences of images
    image_sequences = []

    for i in range(num_sequences):
        # extract sequence of images
        sequence = data[i * seq_len: (i + 1) * seq_len]
        # append sequence to list of image sequences
        image_sequences.append(sequence)

    # check the number of sequences generated
    print(len(image_sequences))
    # check the shape of first sequence generated
    print(image_sequences[0].shape)

    return image_sequences

# Timer function
def convert(seconds):
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    days = seconds // 86400
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d:%02d" % (days, hours, minutes, seconds)


# Function for generating samples
def generate_samples(img_name, label, n_samples, img_path):
    # choose random sequences
    ix = randint(0, len(img_name), n_samples)

    label = np.array(label)
    img_name = np.array(img_name)
    X1, label = img_name[ix], label[ix]

    imgArray = []
    imgArray_LSTM = []
    label_LSTM = []
    for j in range(len(X1)):
        for i in range(len(X1[1-1])):
            FullName = img_path + X1[j,i]
            img = cv2.imread(FullName[0], 1)
            img = Image.fromarray(img.astype(np.uint8))
            loadedImgSize = np.size(img)

            #Crop images to remove top an bottom information boxes (we want to learn from image data and not letters etc. in the info boxes (Unreliable))
            if loadedImgSize[1] == 1080: #the img is 1080x1920
                img = img.crop((0, 165, 1920, 900))
            else: # If not 1080x1920 then do this (assumed they are otherwise 1280x720) Might need universal fix here instead of hardcode defining image resolution.
                img = img.crop((0, 108, 1280, 612))
            img = np.array(img)
            img = cv2.resize(img, (SIZE, SIZE))
            #cv2.imwrite("image.png", img)
            img = img / 255

            #The following can be used to check "final" images pesented to the network. (Not LSTM'ified)
            #cv2.imshow("image",img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            imgArray.append(img)
        imgArray_LSTM.append(imgArray)
        imgArray = []
        label_LSTM.append(label[j,0])
    imgArray_LSTM = np.array(imgArray_LSTM)
    label_LSTM = np.array(label_LSTM)
    return imgArray_LSTM, label_LSTM

def train(Y_trainImg, Y_trainLabel, Y_valImg, Y_valLabel, model, n_epochs=EPOCHS, n_batch=BATCH_SIZE,
          image_path=img_path):
    print('##### Network Information ##### \n Epochs: %s \n Batch Size: %s \n Early Stopping Iterations: %s ' % (
        EPOCHS, BATCH_SIZE, earlyStoppingVar))
    t = time.time()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(Y_trainImg) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # img_path = "D"


    val_loss_arr = []
    train_loss_arr = []
    Xdimensions = []

    epoch_arr = []

    best_val_loss = 1000  # Start high, we want to be low
    valCount = 1

    train_loss_array = []
    val_loss_array = []
    train_accuracy = 0
    val_accuracy = 0
    acc_arr_val = []
    acc_arr_train = []
    n_batch_array = []
    train_loss_akk = 0
    val_loss_akk = 0
    best_val_acc = 0

    j = 1

    for i in range(n_steps):
        # select a batch of real samples
        X_realA, X_realB = generate_samples(Y_trainImg, Y_trainLabel, n_batch, image_path)

        #Train (and test) on batch is: Single gradient update over one batch of samples. Using this allows for full control between each batch as opposed to "model.fit"
        [train_loss, train_acc] = model.train_on_batch(x=X_realA,
                                                       y=X_realB,
                                                       reset_metrics=True)

        train_loss_array.append(train_loss)

        X_realA_val, X_realB_val = generate_samples(Y_valImg, Y_valLabel, n_batch, image_path)


        [val_loss, val_acc] = model.test_on_batch(x=X_realA_val,
                                                  y=X_realB_val,
                                                  reset_metrics=True)
        val_loss_array.append(val_loss)

        train_loss_akk += train_loss
        val_loss_akk += val_loss
        train_accuracy += train_acc
        val_accuracy += val_acc

        if i == 1: #Only the first iteration to fix matrices (this is not important)
            val_loss_arr.append(val_loss_akk / bat_per_epo)
            train_loss_arr.append(train_loss_akk / bat_per_epo)
            Xdimensions.append(j)
            acc_arr_train.append(train_accuracy / bat_per_epo)
            acc_arr_val.append(val_accuracy / bat_per_epo)
            epoch_arr.append(j)
            j = j + 1

        if valCount % earlyStoppingVar == 0:  # Early stopping is triggered
            print('Validation loss has not improved for %.3f iterations  ' % earlyStoppingVar)
            # summarize_performance(i, model, dataset)
            elapsedTime = time.time() - t  # calculate time between now and start of training
            elapsedTime = convert(elapsedTime)
            print('Elapsed training time: %s ' % elapsedTime)

            # Plot of loss vs batches
            pyplot.plot(epoch_arr[1:], train_loss_arr[1:], color='blue', label='Training loss')
            pyplot.plot(epoch_arr[1:], val_loss_arr[1:], color='red', label='Validation loss')
            pyplot.legend()
            pyplot.ylabel('Loss (average pr. epoch)')
            pyplot.xlabel('EPOCHS')
            filename_loss = 'LossCurve_AR.png'
            pyplot.savefig(filename_loss)
            pyplot.close()

            # Plot of accuracy vs epochs
            pyplot.plot(epoch_arr[1:], acc_arr_train[1:], color='green', label='Training accuracy')
            pyplot.plot(epoch_arr[1:], acc_arr_val[1:], color='yellow', label='Validation accuracy')
            pyplot.legend()
            pyplot.ylabel('Accuracy (average pr. epoch)')
            pyplot.xlabel('Epochs')
            filename_accuracy = 'AccuracyCurve_AR.png'
            pyplot.savefig(filename_accuracy)
            pyplot.close()

            # Save losses and accuracies
            np.savetxt("train_loss_arr_AR.csv", train_loss_arr, delimiter=",")
            np.savetxt("val_loss_arr_AR.csv", val_loss_arr, delimiter=",")

            np.savetxt("acc_arr_train_AR.csv", acc_arr_train, delimiter=",")
            np.savetxt("acc_arr_val_AR.csv", acc_arr_val, delimiter=",")

            break #This breaks the entire training loop

        n_batch_array.append(i + 1)
        #Track number of epochs during training
        if (j == 1):
            print("Epoch", j, "/", EPOCHS)

        #Print losses and acc for each update
        print('>%d, train_loss[%.3f] val_loss[%.3f] train_acc[%.3f] val_acc[%.3f]' % (i + 1, train_loss, val_loss, train_acc, val_acc))



        if (i + 1) % (bat_per_epo) == 0: # Summarize performance for every epoch
            if j <= EPOCHS:
                print("Epoch", j, "/", EPOCHS)
            #Calculate averages for plots
            val_loss_arr.append(val_loss_akk / bat_per_epo)
            train_loss_arr.append(train_loss_akk / bat_per_epo)
            epoch_arr.append(j)
            acc_arr_train.append(train_accuracy / bat_per_epo)
            acc_arr_val.append(val_accuracy / bat_per_epo)
            Xdimensions.append(j)

            #Determine if we have a new best model (based on avg val loss) and save if we do (also reset early triggering counter if we do)
            if val_loss_arr[j - 1] < best_val_loss:
                best_val_loss = val_loss_arr[j - 1]
                best_val_acc = acc_arr_val[j - 1]
                filename1 = 'Best_model_AR.h5'
                model.save(filename1)
                print('>Average (pr. epoch) validation loss for current best model: %.3f, Best Accuracy: %.3f, Model Name: %s' % (best_val_loss, best_val_acc, filename1))
                valCount = 0

            val_loss_akk = 0
            train_loss_akk = 0
            train_accuracy = 0
            val_accuracy = 0
            valCount = valCount + 1
            print('Average validation loss have not improved in: %f epoch(s)' % valCount)
            j = j + 1



        if (i + 1) % (n_steps) == 0:  # Everything is done - All epochs was allowed to run through
            # Plot of loss vs epochs at the end
            pyplot.plot(epoch_arr[1:], train_loss_arr[1:], color='blue', label='Training loss')
            pyplot.plot(epoch_arr[1:], val_loss_arr[1:], color='red', label='Validation loss')
            pyplot.legend()
            pyplot.ylabel('Loss')
            pyplot.xlabel('Epochs')
            filename_loss = 'LossCurve_AR.png'
            pyplot.savefig(filename_loss)
            pyplot.close()

            # Plot of accuracy vs epochs at the end
            pyplot.plot(epoch_arr[1:], acc_arr_train[1:], color='green', label='Training accuracy')
            pyplot.plot(epoch_arr[1:], acc_arr_val[1:], color='yellow', label='Validation accuracy')
            pyplot.legend()
            pyplot.ylabel('Accuracy')
            pyplot.xlabel('Epochs')
            filename_accuracy = 'AccuracyCurve_AR.png'
            pyplot.savefig(filename_accuracy)
            pyplot.close()

            #Save losses and accuracies
            np.savetxt("train_loss_arr_AR.csv", train_loss_arr, delimiter=",")
            np.savetxt("val_loss_arr_AR.csv", val_loss_arr, delimiter=",")

            np.savetxt("acc_arr_train_AR.csv", acc_arr_train, delimiter=",")
            np.savetxt("acc_arr_val_AR.csv", acc_arr_val, delimiter=",")

            elapsedTime = time.time() - t  # calculate time between now and start of training
            elapsedTime = convert(elapsedTime)
            print('Elapsed training time: %s ' % elapsedTime)

        i = i + 1


# Select the model to use - new models should be placed here.
def get_model():
    return Publication_cnn_lstm_net(Size=SIZE, IMG_CHANNELS=3, n_classes=n_classes)

# Build the model
model = get_model()
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy',metrics='categorical_accuracy')
model.summary()

plot_model(model, to_file='CNNLSTM_AR.png', show_shapes=True, show_layer_activations=True)

#To accomodate the LSTM layers of the network we must add a dimension turning many individual images into sequences of images.
Y_trainImg_LSTM = LSTM_Prep(Y_trainImg, Seq_len)
Y_trainLabel_LSTM = LSTM_Prep(Y_trainLabel, Seq_len)
Y_valImg_LSTM = LSTM_Prep(Y_valImg, Seq_len)
Y_valLabel_LSTM = LSTM_Prep(Y_valLabel, Seq_len)
Y_testImg_LSTM = LSTM_Prep(Y_testImg, Seq_len)
Y_testLabel_LSTM = LSTM_Prep(Y_testLabel, Seq_len)


#Save data for possible backtrackin (test data to use in testing)
np.save('TrainLabelLSTM_AR.npy', Y_trainLabel_LSTM)
np.save('TestLabelLSTM_AR.npy', Y_testLabel_LSTM)
np.save('ValLabelLSTM_AR.npy', Y_valLabel_LSTM)
np.save('TrainImgLSTM_AR.npy', Y_trainImg_LSTM)
np.save('TestImgLSTM_AR.npy', Y_testImg_LSTM)
np.save('ValImgLSTM_AR.npy', Y_valImg_LSTM)


# Initiate training with the correct inputs. Testdata must be manually moved to the testing algorithm (different code and folder location)
train(Y_trainImg_LSTM, Y_trainLabel_LSTM, Y_valImg_LSTM, Y_valLabel_LSTM, model, EPOCHS, BATCH_SIZE, img_path)
