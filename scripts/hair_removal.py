# hair_removal.py

import os
import cv2
import random
import numpy as np
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

def remove_hair_from_images(input_folder, output_folder, apply_hair_removal=True):
    """
    Processes images from the input folder and saves them to the output folder.
    If apply_hair_removal is True, hair removal is applied to the images.
    If apply_hair_removal is False, the function does nothing.

    :param input_folder: Path to the folder containing input images.
    :param output_folder: Path to the folder where output images will be saved.
    :param apply_hair_removal: Boolean flag to apply hair removal or not.
    """
    if not apply_hair_removal:
        print("Hair removal not applied. Exiting function.")
        return  # Do nothing

    os.makedirs(output_folder, exist_ok=True)
    
    # Get a list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Use tqdm to display a progress bar
    for filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
    
        if image is not None:
            processed_image = remove_hair(image)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)
        else:
            print(f"Warning: Unable to read image {image_path}")

def remove_hair(image):
    """
    Removes hairs from a single image using morphological operations and inpainting.
    
    :param image: Input image as a NumPy array.
    :return: Processed image with hairs removed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply morphological black-hat filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold the blackhat image to get the hair mask
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the original image using the hair mask
    inpainted_image = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    
    return inpainted_image
