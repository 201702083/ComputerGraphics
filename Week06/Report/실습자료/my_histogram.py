import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def my_padding(src, filter):
    (h, w) = src.shape
    if isinstance(filter, tuple):
        (h_pad, w_pad) = filter
    else:
        (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h + h_pad * 2, w + w_pad * 2))
    padding_img[h_pad:h + h_pad, w_pad:w + w_pad] = src

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

    # filter 확인
    # print('<filter>')
    # print(filter)

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


def calc_derivatives(src):
    # calculate Ix, Iy
    sobel_x, sobel_y = get_my_sobel()
    Ix = my_filtering(src, sobel_x)
    Iy = my_filtering(src, sobel_y)
    return Ix, Iy

def show_patch_hist(patch_vector):
    index = np.arange(len(patch_vector))
    plt.bar(index, patch_vector)
    plt.title('orientation histogram')
    plt.show()


def main():
    src = cv2.imread('Lena.png')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    print('get Ix and Iy...')
    Ix, Iy = calc_derivatives(gray)

    print('calculate angle and magnitude')
    angle = np.rad2deg(np.arctan2(Iy, Ix))  ## -180 ~ 180으로 표현
    angle = (angle + 360) % 360  ## 0 ~ 360으로 표현
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    p_size = 32 #patch size
    p_h = 256 #patch height
    p_w = 256 #patch width

    patch = gray[p_h:p_h+p_size, p_w:p_w+p_size]

    cv2.imshow("patch", patch)
    cv2.waitKey()
    cv2.destroyAllWindows()

    patch_ang = angle[p_h:p_h+p_size, p_w:p_w+p_size]
    patch_mag = magnitude[p_h:p_h+p_size, p_w:p_w+p_size]
    angle_range = 10.

    h, w = patch.shape[:2]
    vector_size = int(360 // angle_range)
    vector = np.zeros(vector_size, )
    for row in range(h):
        for col in range(w):
            vector[int(patch_ang[row, col] // angle_range)] += patch_mag[row, col]


    print('angle')
    print(patch_ang)
    print('magnitude')
    print(patch_mag)
    print('vector')
    print(vector)
    print(np.argmax(vector)*angle_range)

    show_patch_hist(vector)


if __name__ == '__main__':
    main()