import cv2
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img=mpimg.imread('1.jpg')


face_detector = dlib.get_frontal_face_detector()
detected_face = face_detector(img, 1)
imgplot = plt.imshow(detected_face)
plt.show()