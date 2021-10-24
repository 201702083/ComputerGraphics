import numpy as np
import cv2

# bilinear interpolation
def my_bilinear(img, x, y):
    '''
    :param img: 값을 찾을 img
    :param x: interpolation 할 x좌표
    :param y: interpolation 할 y좌표
    :return: img[x,y]에서의 value (bilinear interpolation으로 구해진)
    '''
    floorX, floorY = int(x), int(y)

    t, s = x - floorX, y - floorY

    zz = (1 - t) * (1 - s)
    zo = t * (1 - s)
    oz = (1 - t) * s
    oo = t * s

    interVal = img[floorY, floorX, :] * zz + \
               img[floorY, floorX + 1, :] * zo + \
               img[floorY + 1, floorX, :] * oz + \
               img[floorY + 1, floorX + 1, :] * oo

    return interVal


def backward(img1, M):
    '''
    :param img1: 변환시킬 이미지
    :param M: 변환 matrix
    :return: 변환된 이미지
    '''
    h, w, c = img1.shape
    result = np.zeros((h, w, c))

    for row in range(h):
        for col in range(w):
            xy_prime = ???
            xy = ???

            x_ = xy[0, 0]
            y_ = xy[1, 0]

            if x_ < 0 or y_ < 0 or (x_ + 1) >= w or (y_ + 1) >= h:
                continue

            result[row, col, :] = my_bilinear(img1, x_, y_)

    return result


def my_ls(matches, kp1, kp2):
    '''
    :param matches: keypoint matching 정보
    :param kp1: keypoint 정보.
    :param kp2: keypoint 정보2.
    :return: X : 위의 정보를 바탕으로 Least square 방식으로 구해진 Affine
                변환 matrix의 요소 [a, b, c, d, e, f].T
    '''
    A = []
    B = []
    for idx, match in enumerate(matches):
        ???

    A = np.array(A)
    B = np.array(B)

    try:
        ???
    except:
        print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
        X = None
    return X


def get_matching_keypoints(img1, img2, keypoint_num):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: 추출한 keypoint의 수
    :return: img1의 특징점인 kp1, img2의 특징점인 kp2, 두 특징점의 매칭 결과
    '''
    sift = cv2.xfeatures2d.SIFT_create(keypoint_num)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher_create(cv2.DIST_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    """
    matches: List[cv2.DMatch]
    cv2.DMatch의 배열로 구성

    cv2.DMatch: 
        trainIdx: img1의 kp1, des1에 매칭되는 index
        queryIdx: img2의 kp2, des2에 매칭되는 index

    kp1[queryIdx]와 kp2[trainIdx]는 매칭된 점
    """
    return kp1, kp2, matches


def feature_matching(img1, img2, keypoint_num=None):
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)

    X = my_ls(matches, kp1, kp2)

    M = np.array([[X[0][0], X[1][0], X[2][0]],
                  [X[3][0], X[4][0], X[5][0]],
                  [0, 0, 1]])

    result = backward(img1, M)
    return result.astype(np.uint8)


def main():
    src = cv2.imread('Lena.png')
    src2 = cv2.imread('LenaFaceShear.png')

    result = feature_matching(src, src2)
    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
