import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#median filter
# read the image
image = cv2.imread ('images/cat_n10.jpg'  )
# apply the 3x3 median filter on the image
processed_image = cv2.medianBlur ( image , 3 )
# display image
cv2.imshow ( 'Median Filter Processing' , processed_image )
# save image to disk
cv2.imwrite ( 'images/noise_free.jpg' , processed_image )

# mean filter
# read the image
image2 = cv2.imread('images/GaussianNoise.jpg')
# apply the 3x3 mean filter on the image
kernel = np.ones((3,3),np.float32)/9
processed_image = cv2.filter2D(image2,-1,kernel)
# display image
cv2.imshow('Mean Filter Processing', processed_image)
# save image to disk
cv2.imwrite('images/noisy_out.png', processed_image)


cv2.waitKey ( 0 )