## A24
## S5 APP4

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def load_img_gray(path):
    #plt.gray()
    img_gray = mpimg.imread(path)
    return img_gray


def load_img_color(path):
    #plt.gray()
    img_color = mpimg.imread(path)
    return np.mean(img_color, -1)


def noise_remove(image):
    img_line = image.transpose()
    #img_line = image.ravel()
    #img_line = image

    zero_arr = [ 0.9 * np.exp(1j*np.pi/2) , 0.9 * np.exp(-1j*np.pi/2),  0.95 * np.exp(1j*np.pi/8) , 0.95 * np.exp(-1j*np.pi/8) ]
    pole_arr = [0, -0.99, -0.99, 0.8]

    z0 = np.poly(zero_arr)
    p0 = np.poly(pole_arr)

    img_filtered = signal.lfilter(p0, z0, image)
    img_filtered = np.real(img_filtered)

    return img_filtered
    #plt.gray()
    #imgplot = plt.imshow(img_filtered); plt.title('img filtered')
    #plt.show()


def rotate_image(image):
    size_y = len(image)
    size_x = len(image[0])
    img_rot = np.zeros((size_y, size_x))
    rot_matrix = [[0, 1], [-1, 0]]
    for yy in range( size_y ): # Y scan
        for xx in range( size_x ): # X scan
            e = [xx-size_x+1, size_y-yy-1] # convertion to e base
            u = np.dot(rot_matrix, e) # apply rotation
            p = [u[0], size_y-1 - u[1]] # convertion to file format base
            img_rot[p[1]][p[0]] = image[yy][xx]

    return img_rot

def freq_filter(image):

    result = np.zeros([len(image),len(image[0])])

    b = [1 ,2 ,1]
    a = [2.3914,1.1072,0.5014]


    result = signal.lfilter(b,a,image)
    return result


if __name__ == '__main__':

    test = np.arange(16).reshape(4, 4)
    test2 = test.transpose()
    #test3 = np.reshape(test2, [4, 4])

    #img = load_img_gray('images/goldhill.png')
    img = np.load('images/image_complete.npy')
    print(len(img))
    print(len(img[0]))
    #img = img[0:200]
    img_cleaned = noise_remove(img)
    img_rotated = rotate_image(img_cleaned)
    img_filtered = freq_filter(img_rotated)
    plt.gray()
    plt.subplot(2, 2, 1); plt.title('source image')
    plt.imshow(img)
    plt.subplot(2, 2, 2); plt.title('denoised image')
    plt.imshow(img_cleaned)
    plt.subplot(2, 2, 3); plt.title('image rotated')
    plt.imshow(img_rotated)
    plt.subplot(2, 2, 4); plt.title('image filtered')
    plt.imshow(img_filtered)
    plt.show()


