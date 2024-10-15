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

def freq_filter(image):
    img_out = np.zeros([len(image),len(image[0])])

    b = [1 ,2 ,1]
    a = [2.3914,1.1072,0.5014]

    img_out = signal.lfilter(b,a,image)
    return img_out

def freq_filter_with_python(image):
    img_out = np.zeros([len(image),len(image[0])])

    wp = 500
    fs = 1600

    # buttord
    order, _ = signal.buttord(wp,750,0.2,60,fs)
    print('Butter : ' ,order)
    b,a = signal.butter(order,wp,fs=fs)

    # cheb1ord
    order, _ = signal.cheb1ord(wp,750,0.2,60,fs)
    print('Cheb1 : ' ,order)

    # cheb2ord
    order, _ = signal.cheb2ord(wp,750,0.2,60,fs)
    print('Cheb2 : ' ,order)

    # ellipord
    order, wn = signal.ellipord(wp,750,0.2,60,fs)
    print('Ellip : ' ,order)
    b,a = signal.ellip(order,rp=0.2,rs=60,Wn=wn,fs=fs)

    img_out = signal.lfilter(b,a,image)
    return img_out

def freq_filter_auto(image):
    fs = 1600 # sampling rate
    wp = 500 # pass freq
    ws = 750 # stop freq
    gpass = 0.2
    gstop = 60
    n_order = 100
    b = []
    a = []
    n_butter, wn_butter = signal.buttord(wp, ws, gpass, gstop, fs=fs)
    print('n_butter: ' + str(n_butter))
    if n_butter < n_order:
        n_order = n_butter
        b, a = signal.butter(n_butter, wp, 'low', fs=fs)

    n_cheb1, wn_cheb1 = signal.cheb1ord(wp, ws, gpass, gstop, fs=fs)
    print('n_cheb1: ' + str(n_cheb1))
    if n_cheb1 < n_order:
        n_order = n_cheb1
        filt_type = 'cheb1'
        b, a = signal.cheby1(n_cheb1, gpass, wp, 'low', fs=fs)

    n_cheb2, wn_cheb2 = signal.cheb2ord(wp, ws, gpass, gstop, fs=fs)
    print('n_cheb2: ' + str(n_cheb2))
    if n_cheb2 < n_order:
        n_order = n_cheb2
        filt_type = 'cheb2'
        b, a = signal.cheby2(n_cheb2, gstop, wp, 'low', fs=fs)

    n_ellip, wn_ellip = signal.ellipord(wp, ws, gpass, gstop, fs=fs)
    print('n_ellip: ' + str(n_ellip))
    if n_ellip < n_order:
        n_order = n_ellip
        filt_type = 'ellip'
        b, a = signal.ellip(n_ellip, gpass, gstop, wp, 'low', fs=fs)


    img_out = signal.lfilter(b, a, image)
    print(n_order)

    return img_out


def compress_image(image, percent):
    img_cov = np.cov(image)
    ident = np.identity(len(img_cov))

    e_val, e_vect = np.linalg.eig(img_cov)

    e_sorted = [x for _, x in sorted(zip(e_val, e_vect))]

    img_out = img_cov # TEMP
    return img_out


def compression(image):
    result = np.zeros([len(image),len(image[0])])

    

    return result


if __name__ == '__main__':

    test = np.arange(16).reshape(4, 4)
    test2 = test.transpose()
    #test3 = np.reshape(test2, [4, 4])

    #img = load_img_gray('images/goldhill.png')
    img = np.load('images/image_complete.npy')
    #print(len(img))
    #print(len(img[0]))
    #img = img[0:200]
    img_cleaned = noise_remove(img)
    img_rotated = rotate_image(img_cleaned)
    img_filtered1 = freq_filter(img_rotated)
    img_filtered = freq_filter_auto(img_rotated)

    plt.gray()
    plt.subplot(1, 2, 1); plt.title('Laurent filter')
    plt.imshow(img_filtered1)
    plt.subplot(1, 2, 2); plt.title('Philippe filter')
    plt.imshow(img_filtered)
    plt.show()

    img_compress70 = compress_image(img_filtered, 70)
    img_compress50 = compress_image(img_filtered, 50)

    plt.gray()
    plt.subplot(2, 3, 1); plt.title('source image')
    plt.imshow(img)
    plt.subplot(2, 3, 2); plt.title('filtered image')
    plt.imshow(img_cleaned)
    plt.subplot(2, 3, 3); plt.title('image rotated')
    plt.imshow(img_rotated)
    plt.subplot(2, 3, 4); plt.title('noise removed')
    plt.imshow(img_filtered)
    plt.subplot(2, 3, 4); plt.title('noise removed')
    plt.imshow(img_filtered)
    plt.subplot(2, 3, 5); plt.title('compressed 70%')
    plt.imshow(img_compress70)
    plt.subplot(2, 3, 6); plt.title('compressed 50%')
    plt.imshow(img_compress50)
    plt.show()


