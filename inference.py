import os
import sys
import csv
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np
from keras import backend as K


data_dir = 'C:/ships'
test_dir = os.path.join(data_dir, 'test_v2')


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

dependencies = {'dice_coef': dice_coef}

def gen_pred(test_dir, img, model):
    rgb_path = os.path.join(test_dir,img)
    img = cv2.imread(rgb_path)
    img = img[::3, ::3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(rgb_path), pred


# Function to perform the inference on image samples in a directory
def perform_inference(directory):
    try:
        # List all image files in the directory with common formats (PNG, JPG/JPEG)
        image_files = [file for file in os.listdir(directory) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Create a directory to save the predicted values
        save_directory = os.path.join(directory, 'predicted_values')
        os.makedirs(save_directory, exist_ok=True)
        loaded_model = tf.keras.models.load_model(r'C:\ships\night_model_new_dice.h5',
                                                  custom_objects=dependencies)  # trained_model

        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            img, pred = gen_pred(image_path, loaded_model)

            # Save the predicted value to a file
            save_path = os.path.join(save_directory, f"{image_file}.txt")
            with open(save_path, 'w') as file:
                file.write(str(pred))

        print("Inference completed.")
    except Exception as ex:
        print("The directory path must not contain spaces.")
        print(ex)

# Check if the directory path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the directory path as a command-line argument.")
    sys.exit(1)

# Get the directory path from the command-line argument
directory_path = sys.argv[1]

# Perform inference on the image samples in the directory
perform_inference(directory_path)
