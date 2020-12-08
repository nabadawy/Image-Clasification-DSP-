
import cv2
import numpy as np

#median filter
# read the image
#image = cv2.imread ('images/cat_n10.jpg'  )
# apply the 3x3 median filter on the image
#processed_image = cv2.medianBlur ( image , 3 )
# display image
#cv2.imshow ( 'Median Filter Processing' , processed_image )
# save image to disk
#cv2.imwrite ( 'images/noise_free.jpg' , processed_image )

# Median Spatial Domain Filtering

# Read the image
# Median Spatial Domain Filtering


# Read the image
img_noisy1 = cv2.imread ( 'images/random_low.png',0 )

# Obtain the number of rows and columns
# of the image
m , n = img_noisy1.shape

# Traverse the image. For every 3X3 area,
# find the median of the pixels and
# replace the ceter pixel by the median
img_new1 = np.zeros ( [ m , n ] )

for i in range ( 1 , m - 1 ) :
    for j in range ( 1 , n - 1 ) :
        temp = [ img_noisy1 [ i - 1 , j - 1 ] ,
                 img_noisy1 [ i - 1 , j ] ,
                 img_noisy1 [ i - 1 , j + 1 ] ,
                 img_noisy1 [ i , j - 1 ] ,
                 img_noisy1 [ i , j ] ,
                 img_noisy1 [ i , j + 1 ] ,
                 img_noisy1 [ i + 1 , j - 1 ] ,
                 img_noisy1 [ i + 1 , j ] ,
                 img_noisy1 [ i + 1 , j + 1 ] ]

        temp = sorted ( temp )
        img_new1 [ i , j ] = temp [ 4 ]

img_new1 = img_new1.astype ( np.uint8 )
img= cv2.medianBlur ( img_noisy1 , 3 )
cv2.imwrite ( 'images/new_median_filtered.jpg' , img )

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