import numpy as np
import PIL.Image as image
import matplotlib.pyplot as plt
import cv2, sys
import matplotlib.animation as anime
import scipy.ndimage as ndim
import os, sys
sys.path.append(os.getcwd())
import glob
import use
import test
import time
import datetime

"""

Written by Rocky Li @ Duke Robotics

2018 - 04 - 18

This is a module capable of detecting differences in array by triple frame difference method.
Input could be a list of files or a movie.
Output could be a list of arrays or a movie.

2018 - 04 - 30

Edited to incorporate continuous image intake through files

"""

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

    def __init__(self, colored = False, threshold = 122):
        self.colored = colored
        self.thresh = threshold

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
        mask = diffarray > self.thresh
        return mask

    def findmoving(self):
        mask = np.zeros(self.dimension, dtype=bool)
        pre_mask = mask + self.diff(self.array1, self.array2)
        aft_mask = mask + self.diff(self.array2, self.array3)
        true_mask = np.logical_and(pre_mask, aft_mask)
        # true_mask = self.getsuremask(true_mask)
        # print(true_mask)
        true_mask = np.logical_not(true_mask)
        masked = np.ma.array(self.array2, mask=true_mask, fill_value=0)
        moving = masked.filled()
        return moving

    def getsuremask(self, mask):
        # print(mask)
        counts, clusters = ndim.measurements.label(mask)
        # print(counts, clusters)
        labels=np.arange(clusters)
        labels += 1
        sizes = ndim.measurements.sum(mask, counts, labels)
        print(labels, len(labels))
        print(sizes, len(sizes))
        counts = counts.astype(np.float64)
        for index, size in zip(labels, sizes):
            size -= 0.5
            counts[counts == index] = size
        counts = counts > 512
        return counts

# Package video processing into a nice little class
class identify:

    def __init__(self, generator, name = 'moving'):
        self.generator = generator
        self.name = name
        self.fdiff = framediff(threshold = 2)
        self.setup()
        self.dimension = self.fdiff.dimension
        self.figure = plt.figure(figsize = (3,5))
        self.ax = plt.Axes(self.figure, [0,0,1,1])
        self.ax.set_axis_off()
        self.figure.add_axes(self.ax)
        self.index = 1

    def setup(self):
        startarray = []
        i = 0
        while(True):
            my_pic = next(self.generator)
            if my_pic is False:
                time.sleep(0.8)
                continue
            else:
                startarray.append(my_pic)
                i += 1
            if i == 3:
                break
        self.fdiff.set_array(*startarray)

    def identify(self):
        moving = self.fdiff.findmoving()
        try:
            while(True):
                nextframe = next(self.generator)
                if nextframe is False:
                    time.sleep(0.5)
                    continue
                else:
                    break
            # print('PHP')
        except Exception as e:
            print(e)
            # print('WHA')
            sys.exit('IDK')
        self.fdiff.update_array(nextframe)
        # print(np.sum(moving))
        # Todo: edit moving

        # THIS IS LUKES USE.PY
        moving = use.find_subject(moving)
        if not moving[1][0] == 0:
            if not os.path.isdir('target'):
                os.mkdir('target')
            np.save('target/%05d' % (self.index) , moving[0])
            self.index += 1
        return moving[0]

### -----------    THIS TURNS THE WHOLE THING INTO A MOVIE     -------------###

    def start(self):
        self.myart = self.ax.imshow(self.fdiff.array1, 'gray')

    def update(self, i):
        print(i)
        moving = self.identify()
        self.myart.set_data(moving)
        return self.myart

    def animate(self):
        ani = anime.FuncAnimation(self.figure, self.update, init_func=self.start, frames=500, interval = 250)
        ani.save('%s.mp4' % self.name, fps=5, dpi = 120, bitrate=-1)

### -------------------  TEXT OUTPUT CONTINUOUS -------------------------###

    def text(self):
        while(True):
            self.identify()


# If processing images, put things as a generator
class loadimages:
    #
    def __init__(self, path, csv = False):
        self.path = path
        self.get_image()
        self.csv = csv

    # Set a stream of image.
    def set_image(self, image):
        self.image_stream = image

    # Tool to get stream of images
    def get_image(self):
        images = glob.iglob(self.path)
        # print(glob.glob(self.path))
        self.set_image(images)

    def stream(self):
        while(True):
            curr_image = next(self.image_stream)
            # print(curr_image)
            if self.csv:
                curr_frame = np.genfromtxt(curr_image, delimiter=',')
                curr_frame = curr_frame[:, :-1]
                # print(curr_frame)
            else:
                curr_frame = cv2.imread(curr_image,0)
            yield curr_frame


# This is to generate images on the fly
class yield_images:

    def __init__(self, path, csv= True):
        self.path = path
        self.csv = csv
        self.previmage = ''
        print("Starting yield image process")

    def get_images(self):
        # This is all file within that folder
        images = glob.glob(self.path)
        if len(images) == 0:
            return False
        latest_image = max(images, key=os.path.getctime)
        if len(images) > 100:
            while(len(images) > 100):
                earliest = min(images, key=os.path.getctime)
                earliestpng = earliest.split('.')[0] + '.png'
                os.remove(earliest)
                os.remove(earliestpng)
            # print("Removing old image")
        if latest_image == self.previmage:
            print("Waiting for input")
            return False
        # print("New image added to generator")
        self.previmage = latest_image
        print(latest_image)
        return latest_image

    def stream(self):
        while(True):
            curr_image = self.get_images()
            if curr_image is False:
                yield False
                continue
            if self.csv:
                curr_frame = np.genfromtxt(curr_image, delimiter=',')
                curr_frame = curr_frame[:, :-1]
            else:
                curr_frame = cv2.imread(curr_image,0)
            yield curr_frame


if __name__ == '__main__':
    filename = sys.argv[1]
    # Movie engine on!
    images = loadimages('%s/*.txt' % filename, csv=True).stream()
    # images = yield_images('testfiles/*.txt').stream()
    # frames = frame_gen('long.mp4').framegen(skipframe=2)
    moving = identify(images, name='test3')
    # while(True):
    #     print(moving.identify())
    moving.text()
