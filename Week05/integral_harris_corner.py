import sys

import numpy as np
import cv2
import time


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


def find_local_maxima(src, ksize):
    (h, w) = src.shape
    pad_img = np.zeros((h + ksize, w + ksize))
    pad_img[ksize // 2:h + ksize // 2, ksize // 2:w + ksize // 2] = src
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            max_val = np.max(pad_img[row: row + ksize, col:col + ksize])
            if max_val == 0:
                continue
            if src[row, col] == max_val:
                dst[row, col] = src[row, col]

    return dst


def get_integral_image(src):
    assert len(src.shape) == 2
    h, w = src.shape
    dst = np.zeros(src.shape)
    for row in range(h):
        dst[row, 0] = np.sum(src[0:row + 1, 0])
    for col in range(w):
        dst[0, col] = np.sum(src[0, 0:col + 1])
    for row in range(1, h):
        for col in range(1, w):
            dst[row, col] = src[row, col] + dst[row - 1, col] + dst[row, col - 1] - dst[row - 1, col - 1]

    return dst


def calc_local_integral_value(src, left_top, right_bottom):
    assert len(left_top) == 2
    assert len(right_bottom) == 2

    lt_row, lt_col = left_top
    rb_row, rb_col = right_bottom

    lt_val = src[lt_row - 1, lt_col - 1]
    rt_val = src[lt_row - 1, rb_col]
    lb_val = src[rb_row, lt_col - 1]
    rb_val = src[rb_row, rb_col]

    if lt_row == 0:
        lt_val = 0
        rt_val = 0

    if lt_col == 0:
        lt_val = 0
        lb_val = 0
    # print(lt_val , ' - ', lb_val , ' - ', rt_val , ' + ', rb_val)
    return lt_val - lb_val - rt_val + rb_val


def calc_M_harris(IxIx, IxIy, IyIy, fsize=5):
    assert IxIx.shape == IxIy.shape and IxIx.shape == IyIy.shape
    h, w = IxIx.shape
    M = np.zeros((h, w, 2, 2))
    IxIx_pad = my_padding(IxIx, (fsize, fsize))
    IxIy_pad = my_padding(IxIy, (fsize, fsize))
    IyIy_pad = my_padding(IyIy, (fsize, fsize))
    # N = np.zeros((h,w,2,2))
    # for row in range(h):
    #     for col in range(w):
    #         N[row, col, 0, 0] = np.sum(IxIx_pad[row:row+fsize, col:col+fsize])
    #         N[row, col, 0, 1] = np.sum(IxIy_pad[row:row+fsize, col:col+fsize])
    #         N[row, col, 1, 0] = M[row, col, 0, 1]
    #         N[row, col, 1, 1] = np.sum(IyIy_pad[row:row+fsize, col:col+fsize])

    ##########################################################################
    # ToDo
    # integral을 사용하지 않고 covariance matrix 구하기
    # 4중 for문을 채워서 완성하기
    # 실습 시간에 했던 내용을 생각하면 금방 해결할 수 있음
    ##########################################################################
    for row in range(h):
        for col in range(w):
            for f_row in range(fsize):
                for f_col in range(fsize):
                    M[row, col, 0, 0] += IxIx_pad[row + f_row, col + f_col]
                    M[row, col, 0, 1] += IxIy_pad[row + f_row, col + f_col]
                    M[row, col, 1, 1] += IyIy_pad[row + f_row, col + f_col]
            M[row, col, 1, 0] = M[row, col, 0, 1]

    return M


## Integral을 사용하지 않은 Harris detector
def harris_detector(src, k=0.04, threshold_rate=0.01, fsize=5):
    harris_img = src.copy()
    h, w, _ = src.shape
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) / 255.
    # calculate Ix, Iy
    Ix, Iy = calc_derivatives(gray)  # x y 축 변화량

    # Square of derivatives
    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = Ix * Iy

    start = time.perf_counter()  # 시간 측정 시작
    M_harris = calc_M_harris(IxIx, IyIy, IxIy, fsize)
    end = time.perf_counter()  # 시간 측정 끝
    print('M_harris time : ', end - start)

    R = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            ##########################################################################
            # ToDo
            # det_M 계산
            # trace_M 계산
            # R 계산 Harris 방정식 구현
            ##########################################################################

            # IxIx*IyIy - IxIy^2
            det_M = M_harris[row, col, 0, 0] * M_harris[row, col, 1, 1] - M_harris[row, col, 0, 1] ** 2

            # IxIx + IyIy
            trace_M = M_harris[row, col, 0, 0] + M_harris[row, col, 1, 1]

            R[row, col] = det_M - k * (trace_M * trace_M)
    # thresholding
    R[R < threshold_rate * np.max(R)] = 0

    R = find_local_maxima(R, 21)
    R = cv2.dilate(R, None)

    harris_img[R != 0] = [0, 0, 255]

    return harris_img, M_harris


def calc_M_integral(IxIx_integral, IxIy_integral, IyIy_integral, fsize=5):
    assert IxIx_integral.shape == IxIy_integral.shape and IxIx_integral.shape == IyIy_integral.shape
    h, w = IxIx_integral.shape
    M = np.zeros((h, w, 2, 2))

    IxIx_integral_pad = my_padding(IxIx_integral, (fsize, fsize))
    IxIy_integral_pad = my_padding(IxIy_integral, (fsize, fsize))
    IyIy_integral_pad = my_padding(IyIy_integral, (fsize, fsize))

    ##########################################################################
    # ToDo
    # integral 값을 이용하여 covariance matrix 구하기
    # calc_local_integral_value를 사용하면 쉽게 구할 수 있음
    ##########################################################################
    window = (fsize // 2) * 2
    for row in range(h):

        for col in range(w):

            M[row, col, 0, 0] = calc_local_integral_value(IxIx_integral_pad, (row, col), (row + window, col + window))

            M[row, col, 0, 1] = calc_local_integral_value(IxIy_integral_pad, (row, col), (row + window, col + window))

            M[row, col, 1, 0] = M[row, col, 0, 1]

            M[row, col, 1, 1] = calc_local_integral_value(IyIy_integral_pad, (row, col), (row + window, col + window))

    return M


## Integral을 사용하는 Harris detector
def harris_detector_integral(src, k=0.04, threshold_rate=0.01, fsize=5):
    harris_img = src.copy()
    h, w, c = src.shape
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) / 255.
    # calculate Ix, Iy
    Ix, Iy = calc_derivatives(gray)

    # Square of derivatives
    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = Ix * Iy

    start = time.perf_counter()  # 시간 측정 시작
    IxIx_integral = get_integral_image(IxIx)
    IxIy_integral = get_integral_image(IxIy)
    IyIy_integral = get_integral_image(IyIy)
    end = time.perf_counter()  # 시간 측정 끝
    print('make integral image time : ', end - start)

    start = time.perf_counter()  # 시간 측정 시작
    ##############################
    # ToDo
    # M_integral 완성시키기
    ##############################
    M_integral = calc_M_integral(IxIx_integral, IxIy_integral, IyIy_integral, fsize)
    end = time.perf_counter()  # 시간 측정 끝
    print('M_harris integral time : ', end - start)

    R = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            ##########################################################################
            # ToDo
            # det_M 계산
            # trace_M 계산
            # R 계산 Harris 방정식 구현
            ##########################################################################

            # IxIx*IyIy - IxIy^2
            det_M = M_integral[row, col, 0, 0] * M_integral[row, col, 1, 1] - M_integral[row, col, 0, 1] ** 2

            # IxIx + IyIy
            trace_M = M_integral[row, col, 0, 0] + M_integral[row, col, 1, 1]
            R[row, col] = det_M - k * (trace_M * trace_M)

    # thresholding
    R[R < threshold_rate * np.max(R)] = 0

    R = find_local_maxima(R, 21)
    R = cv2.dilate(R, None)

    harris_img[R != 0] = [0, 0, 255]

    return harris_img, M_integral


def main():
    src = cv2.imread('../image/zebra.png')  # shape : (552, 435, 3)
    print('start!')
    harris_img, MH = harris_detector(src)
    harris_integral_img, MI = harris_detector_integral(src)
    cv2.imshow('harris_img ' + '201702083', harris_img)
    cv2.imshow('harris_integral_img ' + '201702083', harris_integral_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
