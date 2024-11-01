# Video-based-Action-Recognition-and-Skills-Assessment-Porcine-Models
Deep learning network for action recognition and skills assessment. Including public dataset on porcine models.

The codes are open-source. However, when using the code, please make a reference to our paper and repository.

Instructions:

1) Download the video data and labels from: https://www.synapse.org/Synapse:syn63852421/wiki/
2) Download the .txt files with GDPR-secured information about dataset, participants and labels ('Information_about_dataset.txt', 'Information_about_participants.txt', 'Information_about_labels.txt').
3) Use the algorithm, following the steps below:
   Have all the .py files in one folder, all the videos in another folder, and all the labels in a third folder.
   1) Use 'Publication_main vidoes to frames.py' to split videos into indiviual frames
   2) Use 'Publication_Preprocessing.py' to generate training, validation and test sets.
   3) Use 'Publication_mainTraining_ActionRecognition_LSTM.py' for training your action recognition network.
      Use 'Publication_mainTraining_SkillsAssessment_LSTM.py' for training your skills assessment network.
   4) Use 'Publication_mainTesting_ActionRecognition.py' to test your action recognition network.
   5) Use 'Publication_mainTesting_SkillsAssessment.py' to test your skills assessment network.
   6) Use 'Publication_ ActionRecognition_Cross_Validation.py' to do five-fold cross validation for action recognition.
   7) Use 'Publication_Skills Assessment Cross Validation.py'
   8) Use 'Publication_GradCam_DvS.py' to create GradCAM images of your test results (either action recognition or skills assessmnet).


/Hashemi et. al. 2024
