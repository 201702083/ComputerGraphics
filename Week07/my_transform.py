import numpy as np
import cv2

def transformation_backward(src, M):
    h, w, c = src.shape
    rate = 2
    dst = np.zeros((int(h * rate), int(w * rate), c))

    h_, w_ = dst.shape[:2]

    print("backward calc")
    for row_ in range(h_):
        for col_ in range(w_):
            ???

    dst = ((dst - np.min(dst)) / np.max(dst - np.min(dst)) * 255 + 0.5)  # normalization
    return dst.astype(np.uint8)


def transformation_forward(src, M):
    h, w, c = src.shape
    rate = 2  # 변환을 생각하여 임의로 크기를 키움
    dst = np.zeros((int(h * rate), int(w * rate), c))

    h_, w_ = dst.shape[:2]
    count = dst.copy()

    print("forward calc")
    for row in range(h):
        for col in range(w):
            ???

    dst = (dst / count)
    return dst.astype(np.uint8)


def main():
    src = cv2.imread("Lena.png")
    src = cv2.resize(src, None, fx=0.3, fy=0.3)
    theta = -10

    translation = [[1, 0, 350],
                   [0, 1, 350],
                   [0, 0, 1]]
    rotation = [[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]]
    scaling = [[2, 0, 0],
               [0, 2, 0],
               [0, 0, 1]]

    M = np.dot(np.dot(translation, rotation), scaling)

    forward = transformation_forward(src, M)
    backward = transformation_backward(src, M)

    cv2.imshow("forward", forward)
    cv2.imshow("backward", backward)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()