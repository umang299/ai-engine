
import os
import sys
import pydicom
import cv2 as cv
import numpy as np
from PIL import Image
import scipy.io as sio

path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path)

class ReadImage:
    """
    Class to read image with any extension into RGB image.
    """
    def __init__(self, img_path, dtype):
        self.img_path = img_path
        self.dtype = dtype

    def __is_rgb(self, image):
        """
        Function to check if an image is in RGB format or not.
        """
        if len(image.shape) == 3:
            H, W, _ = image.shape
        else:
            H, W = image.shape
        X, Y = np.random.randint(0, H), np.random.randint(0, W)
        color_channels = image[X][Y]
        max_color_index = np.argmax(color_channels)
        if max_color_index == 0:
            return True
        else:
            return False
        
    def __convert_bgr2rgb(self, image):
        """
        Function to convert image in BGR format to RGB.
        """
        image = cv.cvtColor(cv.convertScaleAbs(image), cv.COLOR_BGR2RGB)
        return image
        
    def __read_png_jpg_gif_bmp(self):
        """
        Function to read images with .png, .jpg, .gif or .bmp extension.
        """
        image = np.array(Image.open(self.img_path), dtype=self.dtype)
        return image
    
    def __read_dicom(self):
        """
        Function to read dicom images.
        """
        image = pydicom.read_file(self.img_path).pixel_array
        return image
    
    def __read_mat_np(self):
        """
        Function to read images with .mat or .np extension.
        """
        image = np.array(sio.loadmat(self.img_path)['data'], self.dtype)
        return image
    
    def __read_data(self):
        """
        Function to read image.
        """
        if self.img_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')):
            image = self.__read_png_jpg_gif_bmp()
        elif self.img_path.endswith(('.mat', '.np')):
            image = self.__read_mat_np()
        elif self.endswith(('.dcm')):
            image = self.__read_dicom()
        else:
            raise ValueError('Unsupported image format')
        return image
  
    def __call__(self):
        """
        Function to read image for a given image path and pixel data type.

        Args:
            img_path (str): Image path.
            dtype (np.dtype): Numpy data type.
        Returns:
            image (np.array): 3D image array.
        """
        image = self.__read_data()
        if self.__is_rgb(image=image):
            return image
        else:
            image = self.__convert_bgr2rgb(image=image)
        return image
    
def read_image(img_path, dtype):
    """
    Function to read image for a given image path and pixel data type.

    Args:
        img_path (str): Image path.
        dtype (np.dtype): Numpy data type.
    Returns:
        image (np.array): 3D image array.
    """
    reader = ReadImage(img_path=img_path, dtype=dtype)
    image = reader()
    return image