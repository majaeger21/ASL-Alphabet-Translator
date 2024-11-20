# ASL-Alphabet-Translator

## collect-hand-imgs.py
    This Python script collects image data for a machine learning dataset by capturing frames
    from a webcam. It organizes the images into folders based on class labels, allowing 
    the user to prepare a dataset with a specified number of classes and images per class.
## create-hands-data.py 
    This script processes a dataset of hand images to extract landmarks using MediaPipe's 
    hand tracking module. It converts the hand landmarks into a flattened array of 
    coordinates, associates them with class labels, and saves the processed data and 
    labels to a pickle file for use in machine learning tasks.
## train-model.py
    This script trains a Random Forest classifier on hand landmark data extracted from images. 
    It splits the data into training and testing sets, evaluates the model's accuracy on the test 
    set, and saves the trained model to a pickle file for future use.
## webcam-predict.py
    This script uses the trained Random Forest model to perform real-time hand gesture 
    recognition via webcam. It leverages MediaPipe's hand-tracking to detect and process 
    hand landmarks, flipping the image when detecting a right hand for proper orientation. 
    The model predicts the gesture class, overlays the classification result on 
    the video feed, and draws a bounding box and hand skeleton around the detected hand.

### How to run:
    1. pip install -r requirements.txt
    2. python3 collect-hand-imgs.py
    3. python3 create-hands-data.py <-- Make sure DATA_DIR = './data'
    4. python3 train-model.py 
    5. python3 webcam-predict1.py
