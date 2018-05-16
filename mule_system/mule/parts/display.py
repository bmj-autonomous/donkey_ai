from parts.base import BasePart
import cv2


class DisplayFeed(BasePart):
    ''' Displays image feed to screen '''
    input_keys = ('camera_array', )
    output_keys = ()

    def __init__(self, window_name):
        self.window_name = window_name

    def start(self):
        ''' Starts persistent window '''
        cv2.namedWindow(self.window_name)

    def transform(self, state):
        ''' Displays image 

            opencv utiliizes BGR format so conversion is necessary

            The 5 millisecond wait is necessary in order for the window
            to display the image
        '''
        cv2.imshow(self.window_name, cv2.cvtColor(state[self.input_keys[0]], cv2.COLOR_RGB2BGR))
        cv2.waitKey(5)


    def stop(self):
        ''' Closes window '''
        cv2.destroyWindow(self.window_name)



