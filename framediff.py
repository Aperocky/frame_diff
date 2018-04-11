import numpy as np
import PIL.Image as image
import matplotlib.pyplot as plt
import cv2, sys
import matplotlib.animation as anime
import skvideo.io as sv

# Process videos as a frame generator.
class frame_gen:

    def __init__(self, filename):
        self.file = filename
        self.cap = cv2.VideoCapture(filename)

    def framegen(self, skipframe = 5):
        while(True):
            for i in range(skipframe):
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
        mask = diffarray > 0.5*255
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
        self.generator = frame_gen(filename).framegen(skipframe = 2)
        self.fdiff = framediff()
        self.setup()
        self.dimension = self.fdiff.dimension
        self.figure = plt.figure(figsize = (3,5))
        self.ax = plt.Axes(self.figure, [0,0,1,1])
        self.ax.set_axis_off()
        self.figure.add_axes(self.ax)

    def setup(self):
        startarray = []
        for i in range(3):
            startarray.append(next(self.generator))
        self.fdiff.set_array(*startarray)

    def identify(self):
        moving = self.fdiff.findmoving()
        try:
            nextframe = next(self.generator)
        except Exception as e:
            sys.exit()
        self.fdiff.update_array(nextframe)
        return moving

    def start(self):
        self.myart = self.ax.imshow(self.fdiff.array1, 'gray')

    def update(self, i):
        print(i)
        moving = self.identify()
        self.myart.set_data(moving)
        return self.myart

    def animate(self):
        ani = anime.FuncAnimation(self.figure, self.update, init_func=self.start, frames=2000, interval = 25)
        ani.save('longchange.mp4', fps=30, dpi = 120, bitrate=-1)

if __name__ == '__main__':
    moving = identify('long.mp4')
    moving.animate()
