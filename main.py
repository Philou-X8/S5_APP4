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
            e = [xx-(size_x-1), (size_y-1)-yy] # convertion to e base
            u = np.dot(rot_matrix, e) # apply rotation
            p = [u[0], (size_y-1)-u[1]] # convertion to file format base
            img_rot[p[1]][p[0]] = image[yy][xx] # transfer pixel

    return img_rot

def freq_filter_manual(image):
    img_freq = image
    return img_freq


def freq_filter_auto(image):
    f_sample = 1600 # sampling rate
    f_pass = 500/(f_sample/2) # pass freq
    f_stop = 750/(f_sample/2) # stop freq
    td = 1
    wp = (2/td)*np.tan(f_pass/2)
    ws = (2/td)*np.tan(f_stop/2)
    gpass = 0.2
    gstop = 60
    fs = 0
    n_order = 100
    b = []
    a = []
    n_butter, wn_butter = signal.buttord(wp, ws, gpass, gstop)
    print('n_butter: ' + str(n_butter))
    if n_butter < n_order:
        n_order = n_butter
        b, a = signal.butter(n_butter, wn_butter, 'low')

    n_cheb1, wn_cheb1 = signal.cheb1ord(wp, ws, gpass, gstop)
    print('n_cheb1: ' + str(n_cheb1))
    if n_cheb1 < n_order:
        n_order = n_cheb1
        filt_type = 'cheb1'
        b, a = signal.cheby1(n_cheb1, wn_cheb1, 'low')

    n_cheb2, wn_cheb2 = signal.cheb2ord(wp, ws, gpass, gstop)
    print('n_cheb2: ' + str(n_cheb2))
    if n_cheb2 < n_order:
        n_order = n_cheb2
        filt_type = 'cheb2'
        b, a = signal.cheby1(n_cheb2, wn_cheb2, 'low')

    n_ellip, wn_ellip = signal.ellipord(wp, ws, gpass, gstop)
    print('n_ellip: ' + str(n_ellip))
    if n_ellip < n_order:
        n_order = n_ellip
        filt_type = 'ellip'
        b, a = signal.cheby1(n_ellip, wn_ellip, 'low')

    # b, a = signal.butter(n_order, wn, 'low', output='ba')
    img_filtered = signal.lfilter(b, a, image)
    print(n_order)
    #plt.plot(wn, )

    plt.gray()
    plt.imshow(image); plt.title('img original')
    plt.show()
    plt.imshow(img_filtered); plt.title('img filtered')
    plt.show()
    return img_filtered


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

    img_denoise = freq_filter_auto(img_rotated)

    plt.gray()
    plt.subplot(2, 3, 1); plt.title('source image')
    plt.imshow(img)
    plt.subplot(2, 3, 2); plt.title('filtered image')
    plt.imshow(img_cleaned)
    plt.subplot(2, 3, 3); plt.title('image rotated')
    plt.imshow(img_rotated)
    plt.subplot(2, 3, 4); plt.title('image high freq removed')
    plt.imshow(img_denoise)
    plt.show()


