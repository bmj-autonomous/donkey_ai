from parts.base import BasePart

import cv2
import numpy as np


# TODO: add processing pipeline as a vehicle part rather than all these individual
#       image transformations
# TODO: reimplement image stacking (this perhaps deserves a separate location as we need
#       set up a queueing mechanism within the part to hold the previous N images

class BaseImageProcessor(BasePart):
    input_keys = ('camera_array',)
    output_keys = ('camera_array',)

    def start():
        pass

    def transform(self, state):
        pass

    def stop()
        pass


class ColorSpaceMapProcessor(BaseImageProcessor):
    ''' Map between colorspaces

        For example: colorspace_map = cv2.COLOR_RGB2BGR
                     colorspace_map = cv2.COLOR_RGB2HSV
    '''
    def __init__(self, colorspace_map):
        self.colorspace_map = colorspace_map

    def transform(self, state):
        state[self.output_keys[0]] = cv2.cvtColor(state[self.input_keys[0]], self.colorspace_map)



class ColorSpaceThresholdProcessor(BaseImageProcessor):
    ''' Threshold colorspaces

        Often done in HSV space to identify certain colors
    '''
    def __init__(self, lower_threshold=(110,50,50), upper_threshold=(130,215,215))
        self.lower_threshold = np.array(lower_threshold)
        self.upper_threshold = np.array(upper_threshold)

    def transform(self, state):
        mask = cv2.inRange(state[self.input_keys[0]], 
                self.lower_threshold, self.upper_threshold)

        state[self.output_keys[0]] = cv2.bitwise_and(state[self.input_keys[0]], 
                state[self.input_keys[0]], mask=mask)


# TODO: add kernel support
class CannyProcessor(BaseImageProcessor):
    ''' Canny edge detection
        
        See: https://docs.opencv.org/3.0-beta/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
    '''

    def __init__(self, lower_threshold=60, upper_threshold=110):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold


    def transform(self, state):
        state[self.output_keys[0]] = cv2.Canny(state[self.input_keys[0]], 
                                          self.lower_threshold, 
                                          self.upper_threshold)



class GaussianBlurProcessor(BaseImageProcessor):

    def __init__(self, kernel=(5,5)):
        self.kernel = kernel

    def transform(self, state):
        state[self.output_keys[0]] = cv2.GaussianBlur(state[self.input_keys[0]], 
                                kernel, 
                                0)



class CropProcessor(BaseImageProcessor):
    """
    Crop an image to an area of interest. 
    """
    def __init__(self, margin=(0,0,0,0)):
        # top bottom left right
        self.margin = margin

    def transform(self, state):
        image_shape = state[self.input_keys[0]].shape[:2]
        height_slice = slice(self.margin[0], image_shape[0] - self.margin[1])
        width_slice = slice(self.margin[2], image_shape[1] - self.margin[3])

        state[self.output_keys[0]] = state[self.input_keys[0]][height_slice, width_slice]

