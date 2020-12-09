
import cv2
import numpy as np

def gauss2D(shape=(3,3),sigma=0.5):

    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h




def rotate (image_in, R='0'):
    if R== '0':
        return image_in
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

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = [image_in[i - 1, j - 1],
                        image_in[i - 1, j],
                        image_in[i - 1, j + 1],
                        image_in[i, j - 1],
                        image_in[i, j],
                        image_in[i, j + 1],
                        image_in[i + 1, j - 1],
                        image_in[i + 1, j],
                        image_in[i + 1, j + 1]]

                temp = sorted(temp)
                img_new1[i, j] = temp[4]

        img_new1 = img_new1.astype ( np.uint8 )
        img_new1 = rotate ( img_new1 , R )
        return img_new1

    if type == 'gaussian' :
        m , n = image_in.shape

        # Develop Averaging filter(3, 3) mask
        mask = gauss2D((3,3), 1)


        # Convolve the 3X3 mask over the image
        img_new = np.zeros ( [ m , n ] )

        for i in range ( 1 , m - 1 ) :
            for j in range ( 1 , n - 1 ) :
                temp = image_in [ i - 1 , j - 1 ] * mask [ 0 , 0 ] + image_in [ i - 1 , j ] * mask [ 0 , 1 ] + \
                       image_in [
                           i - 1 , j + 1 ] * \
                       mask [ 0 , 2 ] + image_in [ i , j - 1 ] * mask [ 1 , 0 ] + image_in [ i , j ] * mask [ 1 , 1 ] + \
                       image_in [
                           i , j + 1 ] * mask [ 1 , 2 ] + image_in [ i + 1 , j - 1 ] * mask [ 2 , 0 ] + image_in [
                           i + 1 , j ] * \
                       mask [
                           2 , 1 ] + image_in [ i + 1 , j + 1 ] * mask [ 2 , 2 ]

                img_new [ i , j ] = temp

        img_new = img_new.astype ( np.uint8 )
        img_new = rotate ( img_new , R )
        return img_new







#################################

#input from the user



imag= input ('The image path: ')
type = input ("Type of the filter: ")
r= input ("Enter the degree of rotation if no rotation needed enter 0 :")

img = cv2.imread(imag,0)
out= image_minp(img, r, type)
cv2.imwrite('images/out.jpg', out)









