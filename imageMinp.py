
import cv2
import numpy as np



def rotate (image_in, R='0'):
    if R == '90' :
        img_rotate_90_clockwise = cv2.rotate ( image_in , cv2.ROTATE_90_CLOCKWISE )
        return img_rotate_90_clockwise
    if R == '270' :
        img_rotate_90_counterclockwise = cv2.rotate ( image_in , cv2.ROTATE_90_COUNTERCLOCKWISE )
        return img_rotate_90_counterclockwise
    if R == '180' :
        img_rotate_180 = cv2.rotate ( image_in , cv2.ROTATE_180 )
        return img_rotate_180


def image_minp(image_in, R= '0' , type='lowpass' ):
    if type == 'lowpass':
        m , n = image_in.shape

        # Develop Averaging filter(3, 3) mask
        mask = np.ones ( [ 3 , 3 ] , dtype = int )
        mask = mask / 9

        # Convolve the 3X3 mask over the image
        img_new = np.zeros ( [ m , n ] )

        for i in range ( 1 , m - 1 ) :
            for j in range ( 1 , n - 1 ) :
                temp = image_in [ i - 1 , j - 1 ] * mask [ 0 , 0 ] + image_in [ i - 1 , j ] * mask [ 0 , 1 ] + image_in [
                    i - 1 , j + 1 ] * \
                       mask [ 0 , 2 ] + image_in [ i , j - 1 ] * mask [ 1 , 0 ] + image_in [ i , j ] * mask [ 1 , 1 ] + image_in [
                           i , j + 1 ] * mask [ 1 , 2 ] + image_in [ i + 1 , j - 1 ] * mask [ 2 , 0 ] + image_in [ i + 1 , j ] * \
                       mask [
                           2 , 1 ] + image_in [ i + 1 , j + 1 ] * mask [ 2 , 2 ]

                img_new [ i , j ] = temp

        img_new = img_new.astype ( np.uint8 )
        img_new= rotate(img_new, R)
        return img_new
    if type == 'median':
        m , n = image_in.shape

        # Traverse the image. For every 3X3 area,
        # find the median of the pixels and
        # replace the ceter pixel by the median
        img_new1 = np.zeros ( [ m , n ] )

        for i in range ( 1 , m - 1 ) :
            for j in range ( 1 , n - 1 ) :
                temp = [ image_in [ i - 1 , j - 1 ] ,
                         image_in [ i - 1 , j ] ,
                         image_in [ i - 1 , j + 1 ] ,
                         image_in [ i , j - 1 ] ,
                         image_in [ i , j ] ,
                         image_in [ i , j + 1 ] ,
                         image_in [ i + 1 , j - 1 ] ,
                         image_in [ i + 1 , j ] ,
                         image_in [ i + 1 , j + 1 ] ]

                temp = sorted ( temp )
                img_new1 [ i , j ] = temp [ 4 ]

        img_new1 = img_new1.astype ( np.uint8 )
        img_new1 = rotate ( img_new1 , R )
        return img_new1

    if type == 'mean':
        kernel = np.ones ( (3 , 3) , np.float32 ) / 9
        processed_image = cv2.filter2D ( image_in , -1 , kernel )
        processed_image = rotate ( processed_image , R )
        return processed_image





#################################

#input from the user


img = cv2.imread('images/cat_n10.jpg',0)
type = input ("Enter the type of the filter:")
r= input ("Enter the degree of rotation if no rotation needed enter 0 :")
# low pass filter
out= image_minp(img, r, type)
cv2.imwrite('images/out.jpg', out)









