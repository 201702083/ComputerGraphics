import numpy as np
import cv2

def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src

    # repetition padding
    # up
    padding_img[:h_pad, w_pad:w_pad + w] = src[0, :]
    # down
    padding_img[h_pad + h:, w_pad:w_pad + w] = src[h - 1, :]
    # left
    padding_img[:, :w_pad] = padding_img[:, w_pad:w_pad + 1]
    # right
    padding_img[:, w_pad + w:] = padding_img[:, w_pad + w - 1:w_pad + w]

    return padding_img

def my_filtering(src, filter):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape

    #filter 확인
    print('<filter>')
    print(filter)

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, filter)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + f_h, col:col + f_w] * filter)

    return dst

def get_my_sobel():
    sobel_x = np.dot(np.array([[1], [2], [1]]), np.array([[-1, 0, 1]]))
    sobel_y = np.dot(np.array([[-1], [0], [1]]), np.array([[1, 2, 1]]))
    return sobel_x, sobel_y

def my_get_Gaussian_filter(fshape, sigma=1):
    (f_h, f_w) = fshape
    y, x = np.mgrid[-(f_h // 2):(f_h // 2) + 1, -(f_w // 2):(f_w // 2) + 1]
    #2차 gaussian mask 생성
    filter_gaus =   1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))
    #mask의 총 합 = 1
    filter_gaus /= np.sum(filter_gaus)
    return filter_gaus

def GaussianFiltering(src, fshape = (3,3), sigma=1):
    gaus = my_get_Gaussian_filter(fshape, sigma)
    dst = my_filtering(src, gaus)
    return dst

def find_local_maxima(src, ksize):
    (h, w) = src.shape
    pad_img = np.zeros((h+ksize, w+ksize))
    pad_img[ksize//2:h+ksize//2, ksize//2:w+ksize//2] = src
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            max_val = np.max(pad_img[row : row+ksize, col:col+ksize])
            if max_val == 0:
                continue
            if src[row, col] == max_val:
                dst[row, col] = src[row, col]

    return dst

def calc_derivatives(src):
    """
    #ToDo
    3x3 sobel 필터를 사용해서 Ix Iy 구하기
    :param src: 입력 이미지 (흑백)
    :return: Ix, Iy
    """
    # calculate Ix, Iy
    Ix = ???
    Iy = ???
    return Ix, Iy


def HarrisDetector(src, gaus_filter_size = 3, gaus_sigma = 1, alpha = 0.04, threshold_rate = 0.01):
    (h, w) = src.shape
    # calculate Ix, Iy
    Ix, Iy = calc_derivatives(src)

    #0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    dst_x_Norm = ((Ix - np.min(Ix) )/np.max(Ix - np.min(Ix)) * 255 + 0.5).astype(np.uint8)
    dst_y_Norm = ((Iy - np.min(Iy) )/np.max(Iy - np.min(Iy)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('dst_x_Norm', dst_x_Norm)
    cv2.imshow('dst_y_Norm', dst_y_Norm)
    cv2.waitKey()
    cv2.destroyWindow('dst_x_Norm')
    cv2.destroyWindow('dst_y_Norm')

    """
    #ToDo
    IxIx = Ix^2
    IyIy = Iy^2
    IxIy = Ix * Iy
    #구하기
    """
    # Square of derivatives
    IxIx = ???
    IyIy = ???
    IxIy = ???

    #0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    dst_IxIy_Norm = ((IxIy - np.min(IxIy) )/np.max(IxIy - np.min(IxIy)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('IxIx', IxIx)
    cv2.imshow('IyIy', IyIy)
    #cv2.imshow('IxIy', IxIy)
    cv2.imshow('dst_IxIy_Norm', dst_IxIy_Norm)
    cv2.waitKey()
    cv2.destroyWindow('IxIx')
    cv2.destroyWindow('IyIy')
    #cv2.destroyWindow('IxIy')
    cv2.destroyWindow('dst_IxIy_Norm')

    # Gaussian filter
    """
    #ToDo
    #가우시안 필터 적용하기
    #G_IxIx = IxIx에 가우시안 필터 적용
    #G_IyIy = IyIy에 가우시안 필터 적용
    #G_IxIy = IxIy에 가우시안 필터 적용    
    """

    G_IxIx = ???
    G_IyIy = ???
    G_IxIy = ???

    #0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    G_dst_IxIy_Norm = ((G_IxIy - np.min(G_IxIy) )/np.max(G_IxIy - np.min(G_IxIy)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('G_IxIx', G_IxIx)
    cv2.imshow('G_IyIy', G_IyIy)
    #cv2.imshow('G_IxIy', G_IxIy)
    cv2.imshow('G_dst_IxIy_Norm', G_dst_IxIy_Norm)
    cv2.waitKey()
    cv2.destroyWindow('G_IxIx')
    cv2.destroyWindow('G_IyIy')
    #cv2.destroyWindow('G_IxIy')
    cv2.destroyWindow('G_dst_IxIy_Norm')

    # ornerness function - both eigenvalues are strong
    """
    #ToDo
    # har 구하기  교수님 이론 pdf 67page 참고
    """
    har = ???

    #0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    G_dst_har_Norm = ((har - np.min(har) )/np.max(har - np.min(har)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('har before threshold', G_dst_har_Norm)

    # thresholding
    har[har < threshold_rate * np.max(har)] = 0

    #0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    G_dst_har_thresh_Norm = ((har - np.min(har) )/np.max(har - np.min(har)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('har after threshold', G_dst_har_thresh_Norm)

    #주변에서 가장 큰 값만 남기고 나머지 지우기
    dst = find_local_maxima(har, 21)

    return dst

def main():
    src = cv2.imread('../image/zebra.png')
    harris_img = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    cv2.imshow('original', src)

    dst = HarrisDetector(gray, gaus_filter_size=3, gaus_sigma=1, alpha = 0.04)
    dst = cv2.dilate(dst, None)
    interest_points = np.zeros((dst.shape[0], dst.shape[1], 3))
    interest_points[dst != 0]=[0, 0, 255]
    harris_img[dst != 0]=[0, 0, 255]
    cv2.imshow('interest points', interest_points)
    cv2.imshow('harris_img', harris_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()
