import cv2
import numpy as np

def simple_denoise(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Apply Gaussian Blur as placeholder for deep autoencoder
    denoised = cv2.GaussianBlur(image, (5, 5), 0)

    return denoised
