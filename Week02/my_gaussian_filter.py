import numpy as np
import cv2
import time

def my_padding(src, kernel):
    (h, w) = src.shape
    (h_pad, w_pad) = kernel.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    return padding_img

def my_filtering(src, kernel):
    (h, w) = src.shape
    (k_h, k_w) = kernel.shape
    pad_img = my_padding(src, kernel)
    dst = np.zeros((h, w)) #output

    for m in range(h):
        for n in range(w):
            sum = 0
            for k in range(k_h):
                for l in range(k_w):
                    sum += kernel[k, l] * pad_img[m + k, n + l]
            dst[m, n] = sum

    dst = (dst + 0.5).astype(np.uint8)
    return dst

def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    '''
    y, x = np.mgrid[-1:2, -1:2]
    y = [[-1,-1,-1],
         [ 0, 0, 0],
         [ 1, 1, 1]]
    x = [[-1, 0, 1],
         [-1, 0, 1],
         [-1, 0, 1]]
    '''
    f = x * x + y * y
    # f = [[2,1,2],
    #      [1,0,1],
    #      [2,1,2]]

    # 2차 gaussian mask 생성
    gaus2D = ((np.math.e) ** (-(f) / 2)) / 2 * np.math.pi

    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)

    return gaus2D

def my_get_Gaussian1D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 1D gaussian filter 만들기
    #########################################
    x = np.full((1, msize), [range(-(msize // 2), (msize // 2) + 1)])

    '''
    x = np.full((1, 3), [-1, 0, 1])
    x = [[ -1, 0, 1]]

    x = np.array([[-1, 0, 1]])
    x = [[ -1, 0, 1]]
    '''

    gaus1D = (np.math.e) ** (-( (x * x) / (2 * sigma * sigma))) / np.math.sqrt(2 * np.math.pi)
    # mask의 총 합 = 1
    gaus1D /= np.sum(gaus1D)
    return gaus1D

if __name__ == '__main__':
    # 본인의 Image path를 맞춰서 넣는 것을 생각할 것
    src = cv2.imread('../image/Lena.png', cv2.IMREAD_GRAYSCALE)

    # Gaussian mask를 생성
    mask_size = 5
    gaus2D = my_get_Gaussian2D_mask(mask_size, sigma = 3)
    gaus1D = my_get_Gaussian1D_mask(mask_size, sigma = 3)

    # Gaussian mask와 image를 filtering하고 결과를 출력
    print('mask size : ', mask_size)
    print('1D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus1D= my_filtering(src, gaus1D.T)
    dst_gaus1D= my_filtering(dst_gaus1D, gaus1D)
    end = time.perf_counter()  # 시간 측정 끝
    print('1D time : ', end-start)

    print('2D gaussian filter')
    start = time.perf_counter()  # 시간 측정 시작
    dst_gaus2D= my_filtering(src, gaus2D)
    end = time.perf_counter()  # 시간 측정 끝
    print('2D time : ', end-start)

    dst_gaus1D = np.clip(dst_gaus1D+0.5, 0, 255)
    dst_gaus1D = dst_gaus1D.astype(np.uint8)
    dst_gaus2D = np.clip(dst_gaus2D+0.5, 0, 255)
    dst_gaus2D = dst_gaus2D.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('1D gaussian img', dst_gaus1D)
    cv2.imshow('2D gaussian img', dst_gaus2D)
    cv2.waitKey()
    cv2.destroyAllWindows()