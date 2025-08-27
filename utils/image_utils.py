import cv2
import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model input
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to numpy array
    image = np.array(image)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Convert to CHW format
    if len(image.shape) == 3:
        image = image.transpose(2, 0, 1)
    
    return image

def add_noise(image, noise_type='gaussian'):
    """
    Add noise to image for data augmentation
    """
    if noise_type == 'gaussian':
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss
        return np.clip(noisy, 0, 1)
    elif noise_type == 's&p':
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    else:
        return image

def adjust_lighting(image, factor):
    """
    Adjust image lighting
    """
    image = image * factor
    return np.clip(image, 0, 1)

def random_crop(image, crop_size=(200, 200)):
    """
    Random crop image
    """
    h, w = image.shape[:2]
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    
    if len(image.shape) == 3:
        return image[top:bottom, left:right, :]
    else:
        return image[top:bottom, left:right]
