Muscle Dystrophy Detection Using Voice
This project aims to detect Muscle Dystrophy severity levels from voice samples using a deep learning approach. It involves extracting MFCC features from audio files, training a CNN + LSTM model, and using this trained model to predict muscle dystrophy severity levels.

Features
Feature Extraction: Uses MFCC (Mel-frequency cepstral coefficients) to convert audio files into numerical features suitable for the model.
Model Architecture: Combines CNN (Convolutional Neural Networks) to extract spatial features from the MFCC spectrogram and LSTM (Long Short-Term Memory) networks to capture temporal dependencies in the speech data.
Prediction: The trained model can predict Muscle Dystrophy severity based on voice data.
Evaluation: Provides accuracy, confusion matrix, and classification report for model performance.
GUI Interface: A user-friendly GUI to allow users to upload an audio file and get the severity prediction.

Installation and Setup

1. Clone or Download the Project
Start by cloning or downloading the repository to your local machine.

2. Install Dependencies
The project requires several Python libraries. You can install all of them using the requirements.txt file:


pip install -r requirements.txt
This will install the following dependencies:

tensorflow: For building and training the deep learning model.
librosa: For extracting audio features (MFCCs).
numpy: For numerical operations.
pandas: For handling datasets.
scikit-learn: For machine learning utilities like evaluation metrics.
matplotlib & seaborn: For plotting graphs like the confusion matrix.
tkinter: For building a graphical user interface (GUI).
3. Dataset
The project requires a dataset of audio files with different levels of Muscle Dystrophy severity. The dataset should be organized as follows:

train/: Contains subfolders for each severity level with training audio samples.
test/: Contains subfolders for each severity level with testing audio samples.
validation/: Contains subfolders for each severity level with validation audio samples.
You can either download or collect an appropriate dataset, ensuring it is in a format compatible with the MFCC feature extraction. If you do not have a dataset, you can explore available datasets related to speech disorders or use any audio dataset with voice recordings.

4. Training the Model
Once you have the dataset in the dataset/ folder, you can start training the model. To do so, run the following command:

python code/train_model.py
This will:

Extract MFCC features from the audio files.
Train a CNN + LSTM model on these features.
Save the trained model as muscle_dystrophy_model.h5.
5. Model Evaluation
After training, you can evaluate the model's performance using the following script:

python code/evaluate_model.py
This script will:

Evaluate the model on the test data.
Display performance metrics, including accuracy, loss, confusion matrix, and classification report.
6. Making Predictions
To make predictions on new audio files, run the following script:

python code/make_predictions.py
Alternatively, if you want a graphical user interface (GUI) for uploading audio and viewing predictions, use the GUI.py script:

python code/GUI.py
This will open a window where you can upload an audio file, and the model will predict the severity of Muscle Dystrophy.

7. Optional: Demo Audio Files
If you'd like to test the system right away, you can use the demo audio files in the demo/ folder, or replace them with your own audio files for prediction.

Model Details
Architecture: The model combines CNN and LSTM layers for feature extraction and sequence modeling.
CNN is used to process the MFCC spectrograms.
LSTM is used to capture temporal patterns in the audio features.
Loss Function: Sparse categorical crossentropy is used for multi-class classification.
Optimizer: Adam optimizer is used with a learning rate of 0.001.
Activation Function: Softmax is used in the output layer for multi-class classification.
Accuracy: The model is designed to achieve high accuracy on the test set (depending on the quality of the dataset).

License
This project is licensed under the MIT License.