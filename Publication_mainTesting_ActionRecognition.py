'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'### The following script is made as part of the study #Video-based robotic surgical action recognition and skills assessment on porcine models using deep learning###'
'### The code is open-source. However, when using the code, please make a reference to our paper and repository.""""""""""""""""""""""""""""""""""""""""""""""""""""""'
'"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""'


import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

#Loading of trained model to test on.
modelName = 'NAME OF TESTFILE.h5'

model = keras.models.load_model(modelName)
model.summary()

SIZE = 256 #Final image size to test on (must be the same as the model is trained on!)
n_classes = 2

#Location of all individual frames
img_path = r"PATH"

#Loading test data
#Y_testImg = np.load('TestImgLSTM_ProcrineData1_PrimaryAC.npy', allow_pickle=True)
#Y_testLabel = np.load('TestLabelLSTM_ProcrineData1_PrimaryAC.npy', allow_pickle=True)
Y_testImg = np.load('NAME OF TESTIMG.npy', allow_pickle=True)
Y_testLabel = np.load('NAME OF TESTLABEL.npy', allow_pickle=True)


# function for generating and predicting
# The reason for mixing loading and prediction is that it can be problematic to load ALL images (takes up a lot of space) and then predict.
# Instead, images are loaded in sequences and then predicted upon. This slows down the overall testing a little bit, however, it ensures
# that there are no limits to how much data can be predicted upon. (this error did occur when using all of porcrine dataset2 to test on).

def loadNpredict_testData(img_name, label, img_path):
    predictions = []
    probabilities = []
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
            if loadedImgSize[1] == 1080: #the img is 1080x1920
                img = img.crop((0, 165, 1920, 900))
            else:
                img = img.crop((0, 108, 1280, 612))
            img = np.array(img)
            img = cv2.resize(img, (SIZE, SIZE))
            img = img / 255
            imgArray.append(img)
            h =h+1
        sequences = np.array(imgArray)

        #This is the actual prediction process
        pred = np.expand_dims(sequences, axis=0)
        prob = model.predict(pred) #Returns probability! (i.e. n_classes each with one probability which together sums to 1 - because of softmax)
        pred = np.argmax(prob, axis=-1)
        pred = np.expand_dims(pred, axis=-1)
        pred = to_categorical(pred, n_classes) #This returns the actual "winner" (based on the highest probability) i.e. class 1 or class 2 in this case
        predictions.append(pred)
        probabilities.append(prob)

        imgArray_LSTM.append(imgArray)
        imgArray = []
        label_LSTM.append(label[j,0])
    imgArray_LSTM = np.array(imgArray_LSTM)
    label_LSTM = np.array(label_LSTM)
    return imgArray_LSTM, label_LSTM, predictions, probabilities

testImages, testLabels_true, predictions, probabilities = loadNpredict_testData(Y_testImg, Y_testLabel, img_path)

#predictions, probabilities = Prediction(testImages, model, n_classes)
predictions = np.array(predictions)
predictions = np.squeeze(predictions, axis=1)
predictions = predictions.astype(np.int32)


##### Confusion Matrix
targets = ["Suturing", "Dissection"]
y_true = []
y_true_action = []
for i in range(len(testLabels_true)):
    for j in range(n_classes):
        if testLabels_true[i,j] == 1:
            y_true.append(j)
            y_true_action.append(targets[j])

y_pred = []
y_pred_action = []
for i in range(len(predictions)):
    for j in range(n_classes):
        if predictions[i,j] == 1:
            y_pred.append(j)
            y_pred_action.append(targets[j])

confusionMatrix = confusion_matrix(y_true_action, y_pred_action)
cm = ConfusionMatrixDisplay(confusionMatrix, display_labels=targets)
cm.plot()
pyplot.savefig("Confusion_matrix.png")
pyplot.close()

##### Classification report (presicion, recall and F1-score - also accuracy, makro avg, weighted avg and support)
report = classification_report(y_true, y_pred, target_names=targets, digits=3)
with open("Classification_Report.txt", "w") as file:
    file.write(report)

# Find where we have predictions equal to true labels manually (in order to do probability plots
y_true_cat = to_categorical(y_true)
probabilitiesArr = np.array(probabilities)
probabilitiesArr = np.squeeze(probabilitiesArr, axis=1)
# Find the indices of the rows where the two matrices are the same
same_rows_indices = np.nonzero(np.equal(y_true_cat, predictions).all(axis=1))[0]
# Find the indices of the rows where the two matrices are not the same
different_rows_indices = np.nonzero(np.logical_not(np.equal(y_true_cat, predictions).all(axis=1)))[0]

print("Indices of same rows:", same_rows_indices)
print("Indices of different rows:", different_rows_indices)

#Probability plots - Visualisation of the probabilities behind the classifications (shows prob. of both correct and incorrect predicted labels).
prob_row_max = np.amax(probabilitiesArr, axis=1)

# calculate the mean, variance, max and min of the data
mean = np.mean(prob_row_max)
variance = np.var(prob_row_max)
maximum = np.max(prob_row_max)
minimum = np.min(prob_row_max)

mask = np.zeros_like(prob_row_max)
mask[different_rows_indices] = 1
mask_1 = np.zeros_like(prob_row_max, dtype=bool)
mask_1[different_rows_indices] = True

selected = prob_row_max[~mask_1]

# create the plot
fig, ax = plt.subplots()

# create an array of zeros for x-axis values
x = np.zeros_like(prob_row_max[~mask_1])
x = x + 0.01
x_1 = np.zeros_like(prob_row_max[mask == 1])
x_1 = x_1-0.01

# plot the point cloud with all x values set to 0, and label the points
ax.scatter(x, prob_row_max[~mask_1]*100, label='Correct Pred- \nictions ({:.0f})'.format(int(len(prob_row_max[~mask_1]))))
# plot the masked indices using a red dot
ax.scatter(x_1, prob_row_max[mask == 1]*100, label='Incorrect Pred- \nictions ({:.0f})'.format(int(len(prob_row_max[mask == 1]))), color='red')

# plot the variance as error bars, and label the variance error bars
#ax.errorbar(0, mean, fmt='o', color='y', label='Variance: {:.3f}%'.format(variance*100))
# plot the mean value as a point, and label the mean point
ax.axhline(mean*100, color='k', linestyle='--', label='Mean: {:.2f}%'.format(mean*100))

# plot the maximum value as a point, and label the maximum point
ax.axhline(maximum*100, color='g',linestyle='--', label='Maximum: {:.2f}%'.format(maximum*100))

# plot the minimum value as a point, and label the minimum point
ax.axhline(minimum*100, color='m', linestyle='--', label='Minimum: {:.2f}%'.format(minimum*100))


# set the plot title and axis labels
ax.set_title('Probability distribution for predictions of Primary category problem')
ax.set_xlabel('Suturing vs dissection [Sequence]')
ax.set_ylabel('Probability [%]')
# Remove x-axis tick labels
ax.set_xticklabels([])
# Set x-limits
ax.set_xlim([-0.1, 0.1])

# add the legend with the labels specified in the plots
ax.legend(loc='center right')

plt.savefig("DvSProbDist.png")
# show the plot
plt.show()
plt.close()

#Backtrack incorrect labels for further analysis if needed (remember to save to file if important).
incorrectFrames = Y_testImg[different_rows_indices]
incorrectPredLabels = np.array(y_pred)[different_rows_indices]
incorrectTrueLabels = np.array(y_true)[different_rows_indices]


### PROCEDURE PLOTS ###
# Use this code for procedure plots
# THe intervals has to be changed manually.

# yInterval1 = np.repeat(predictions[:,0][0:561],10) #take expert values from prediction in interval frames 10 times
#
# plt.plot(yInterval1)
# plt.xticks(range(0,len(yInterval1),180))# range(start,stop,step)
# plt.yticks(range(0, 1, 1))
# #plt.xticks(range(0,len(yInterval3),50))# range(start,stop,step)
# plt.xlabel("Seconds")
# plt.ylabel("Predictions (Novice = 0 - Experienced = 1")
# plt.title("Experienced")
# plt.savefig("Experienced.png")
# plt.show()

# yInterval2 = np.repeat(predictions[:,0][562:985],10) #take expert values from prediction in interval frames 10 times
# plt.plot(yInterval2) #REMBER TO CHANGE NAME
# plt.xticks(range(0,len(yInterval2),180))# range(start,stop,step)
# plt.yticks(range(0, 1, 1))
# plt.xlabel("Seconds")
# plt.ylabel("Predictions (Novice = 0 - Experienced = 1")
# plt.title("Novice")
# plt.savefig("Novice.png")
# plt.show()
#