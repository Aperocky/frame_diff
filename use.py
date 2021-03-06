import os, sys
sys.path.append(os.getcwd())
import framediff
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from matplotlib import _cntr as cntr

# imagestream = framediff.loadimages('Thermal_Detection_PicBase/*.txt', csv=True).stream()
# movingstream = framediff.identify(imagestream)

"""

Wenchao Zhu @ Duke Robotics

"""

def find_subject(myframe):
    cv2.imwrite('beforetranslating.jpg',myframe)
    # step 1 - copy the current frame picture
    image = myframe
    # np.savetxt('image.txt',image,fmt='%10.2f')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # trying to use the nparray to get the contour, but find it's too much effort

    # then i tried to convert the image to the opencv files
    #print(np.amax(image))
    im = np.array(image * 255 / max(1, np.amax(image)), dtype = np.uint8)
    image = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    # cv2.imwrite('threshed.jpg',image)

    # step 2 - define the filters
    ret, binary = cv2.threshold(image, 10, 256, 0)
    # print(binary)

    # the following steps means it will color the contours on color (000,100,000)
    ret, contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    coloredimage = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(image,contours,0,(0,100,0),3)

    # cv2.imwrite('colored.jpg',image)

    # contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # step 3 - Segmentation
    # kernel = cv2.getStructuringElement(cv2,MORPH_ELLIPSE, (5,5))

    # step 4 - find biggest contours
    # mask_contour = mask.copy()

    # contours, hierarchy = cv2.findContours(mask_contour, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #isolating the largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    detection_string = "Nothing is detected"
    detection_detail = [0,0,0]
    if(contour_sizes!=[]):
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

        # # step 5 - drap the surrounding frame
        (x,y,w,h) = cv2.boundingRect(biggest_contour)
        aspect_ratio = float(w)/h
        # image = color.gray2rgb(image)
        if aspect_ratio >= 2 and w>8 and h>3:
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
            detection_string = "There was a Standing Human"
            detection_detail = [w, h, aspect_ratio]
        if aspect_ratio < 2 and w>4 and h>8:
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
            detection_string = "There was a moving object"
            detection_detail = [w, h, aspect_ratio]
    print()
    print('###### DETECTION AT WORK ######')
    print('\tWE THINK THAT: ')
    print("\twe %s" % detection_string)
    print('#### END OF THIS ITERATION ####')
    print()
    return image, detection_detail

if __name__ == '__main__':
    imagestream = framediff.loadimages('Thermal_Detection_PicBase/*.txt', csv=True).stream()
    movingstream = framediff.identify(imagestream)
    for number in range(0,343):
        myframe = movingstream.identify()
        result = find_subject(myframe)


# final step: print the current picture
