# With-mask
![Screenshot 2023-11-09 185214](https://github.com/sharma-samarth/FaceMaskProject/assets/122161268/ce61deb7-9610-45c3-8925-88675b0c5cac)

# Without-mask
![Screenshot 2023-11-09 185110](https://github.com/sharma-samarth/FaceMaskProject/assets/122161268/5ec22a3a-257d-44f0-9627-8203af6fd0b9)


# Face-Mask Detector
Real time face-mask detection using Deep Learning and OpenCV

## About Project
This project uses a Deep Neural Network, more specifically a Convolutional Neural Network, to differentiate between images of people with and without masks. The CNN manages to get 
an accuracy of **98.2% on the training set** and **97.3% on the test set**. Then the stored weights of this CNN are used to classify as mask or no mask, in real time, using OpenCV.
With the webcam capturing the video, the frames are preprocessed and and fed to the model to accomplish this task. The model works efficiently with no apparent lag time between
wearing/removing mask and display of prediction.

#### The model is capable of predicting multiple faces with or without masks at the same time

### Project Overview

This project is designed to perform face mask detection using Convolutional Neural Networks (CNN) and OpenCV for live face mask detection. The project is implemented in Python and follows the steps outlined below:

### Steps

1. **Import Necessary Libraries:**
   - Importing libraries such as `numpy`, `keras`, `cv2` (OpenCV), and `datetime`.

2. **Build a CNN Model:**
   - A sequential Keras model is created to classify images into two classes: "with mask" and "without mask". The model architecture consists of several convolutional layers, max-pooling layers, and fully connected layers.

3. **Data Preprocessing:**
   - Data augmentation is performed using the `ImageDataGenerator` to augment and preprocess the training and testing image data. It includes rescaling, shear range, zoom range, and horizontal flip.

4. **Load and Train the Model:**
   - The model is trained using the training data (`'train'` and `'test'` directories containing images of individuals with and without masks). The model is trained for 10 epochs using the `fit_generator` method.

5. **Save the Trained Model:**
   - The trained model is saved to a file named `'mymodel.h5'`.

6. **Test Individual Images:**
   - The script loads the saved model and performs inference on an individual image to predict whether the person is wearing a mask or not.

7. **Implement Live Face Mask Detection:**
   - The script captures video from the computer's webcam using OpenCV (`cv2.VideoCapture`).
   - It uses a Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) to detect faces in each frame of the video.
   - For each detected face, it saves the face image as 'temp.jpg' and resizes it to 150x150 pixels.
   - The model is then used to predict whether the person is wearing a mask or not, and the result is displayed on the video frame.
   - The current date and time are also displayed on the frame.
   - The video feed is displayed, and the program continues running until the 'q' key is pressed, at which point the video feed is closed and the program exits.

This project provides real-time monitoring of mask usage in a video feed and combines machine learning (CNN) with computer vision techniques to achieve this task. The trained model is used to make predictions on each face detected in the video stream.




## Dataset

The data used can be downloaded through this [link](https://data-flair.training/blogs/download-face-mask-data/) or can be downloaded from this repository as well (folders 'test' and 
'train'). There are 1314 training images and 194 test images divided into two catgories, with and without mask.

## How to Use

To use this project on your system, follow these steps:

1.Clone this repository onto your system by typing the following command on your Command Prompt:

```
git clone https://github.com/sharma-samarth/FaceMaskProject.git
```
followed by:

```
cd FaceMaskDetector
```

2. Download all libaries using::
```
pip install -r requirements.txt
```

3. Run facemask.py by typing the following command on your Command Prompt:
```
python facemask.py
```

#### The Project is now ready to use !!


