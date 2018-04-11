import numpy as np
import PIL.Image as image
import matplotlib.pyplot as plt
import cv2

# Process videos as a frame generator.
class frame_gen:

    def __init__(self, filename):
        self.file = filename
        self.cap = cv2.VideoCapture(filename)

    def framegen(self, skipframe = 5):
        while(True):
            for i in range(5):
                ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield gray

# Process image
# class preprocess:
#
#     def __init__(self):
#         pass
#
#     def set_image(self, file1, file2):
#         self.file1 = file1
#         self.file2 = file2
#
#     # Load image from source
#     def load_image(self, path, colored = False):
#         pic = image.open(path)
#         if colored:
#             pix = np.asarray(pic)
#         else:
#             pic = pic.convert('LA')
#             pix = np.asarray(pic)[:,:,0]
#         return pix
#
#     def return_arrays(self):
#         pix1 = self.load_image(self.file1)
#         pix2 = self.load_image(self.file2)
#         return pix1, pix2

# Find moving with current frame by consulting the frame before and after.
class framediff:

    def __init__(self, colored = False):
        self.colored = colored
        pass

    def set_array(self, array1, array2, array3):
        self.array1 = array1
        self.array2 = array2
        self.array3 = array3
        self.dimension = array1.shape
        # if len(self.dimension) > 1:
        #     colored = True

    def update_array(self, array):
        self.array1 = self.array2
        self.array2 = self.array3
        self.array3 = array

    def diff(self, array_x, array_y):
        diffarray = np.abs(array_x - array_y)
        mask = diffarray > 0.35*255
        return mask

    def findmoving(self):
        mask = np.zeros(self.dimension, dtype=bool)
        pre_mask = mask + self.diff(self.array1, self.array2)
        aft_mask = mask + self.diff(self.array2, self.array3)
        true_mask = np.logical_and(pre_mask, aft_mask)
        true_mask = np.logical_not(true_mask)
        masked = np.ma.array(self.array2, mask=true_mask, fill_value=0)
        moving = masked.filled()
        return moving

# Package video processing into a nice little class
class identify:

    def __init__(self, filename):
        self.fg = frame_gen(filename)
        self.fdiff = framediff()

    def setup(self):
        



if __name__ == '__main__':
    vimeo = videoprocessor('video.mp4')
    frames = vimeo.framegen()
