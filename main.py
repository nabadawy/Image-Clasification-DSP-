
import matplotlib.pyplot as plt
import numpy as np
import cv2
#from cv2 import cv

#img = plt.imread('images/1.jpg')/float(2**8)


def low_pass_filter(img_in):  # Write low pass filter here
    dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(
    cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img_in.shape
    crow, ccol = rows / 2, cols / 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[int(crow) - 10: int(crow) + 10, int(ccol) - 10: int(ccol) + 10] = 1
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    img_out = img_back
    return img_out

input_image1 = cv2.imread('images/1.jpg', 0)
# Low and high pass filter
output_image1 = low_pass_filter(input_image1)
#succeed2, output_image2 = high_pass_filter(input_image1)
plt.imsave("images/out1.jpg", output_image1, cmap='')
plt.imshow(output_image1, cmap='gray')