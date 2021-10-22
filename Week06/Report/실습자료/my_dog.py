import cv2
import numpy as np

def DoG_pyramids(src, g=5, l=4):
    s = 1. #sigma
    a = l #extrema point
    k = 2. ** (1/a) #step

    sigma = []
    for n in range(4):
        lv_sigma = []
        for l in range(g):
            lv_sigma.append(s * (k ** (n * (g - 1) + l)))
        sigma.append(lv_sigma)

    # size는 기본 이미지에서 1/2배, 1/4배, 1/8배, 1/16배로 진행
    h, w = src.shape
    scales = [
        (h // 2, w // 2),
        (h // 4, w // 4),
        (h // 8, w // 8),
        (h // 16, w // 16)
    ]

    img = cv2.resize(src, dsize=(w//2, h//2), interpolation=cv2.INTER_LINEAR) ## 이미지를 1/2로 재조정

    gaussian_pyramids = [
        np.zeros((scales[0][0], scales[0][1], g)), #1/2
        np.zeros((scales[1][0], scales[1][1], g)), #1/4
        np.zeros((scales[2][0], scales[2][1], g)), #1/8
        np.zeros((scales[3][0], scales[3][1], g)) #1/16
    ]

    print("Gaussian pyramids")
    for s in range(4):
        for j in range(g):
            lv_sigma = sigma[s][j]
            ksize = 2 * int(4 * lv_sigma + 0.5) + 1
            gaussian = cv2.GaussianBlur(img, (ksize, ksize), lv_sigma)
            h, w = scales[s]
            gaussian = cv2.resize(gaussian, dsize=(w, h), interpolation=cv2.INTER_LINEAR) ## gaussian img를 level에 맞게 조정
            gaussian_pyramids[s][:, :, j] = gaussian


    print("calc Diffrence of Gaussian")
    DoG = [
        np.zeros((scales[0][0], scales[0][1], l)),
        np.zeros((scales[1][0], scales[1][1], l)),
        np.zeros((scales[2][0], scales[2][1], l)),
        np.zeros((scales[3][0], scales[3][1], l))
    ]

    for s in range(4):
        for i in range(l):
            prev = gaussian_pyramids[s][:, :, i + 1]
            next = gaussian_pyramids[s][:, :, i]
            DoG[s][:, :, i] = prev - next

    return gaussian_pyramids, DoG


def show_pyramids(src, d=4, h_s=2, w_s=4, name = 'src'):
    assert len(src) == d #동일한 깊이가 아닐 시 에러
    h, w, c = src[0].shape

    assert c == w_s #동일한 채널이 아닐 시 에러
    dst = np.zeros((h * h_s, w * w_s)).astype(np.uint8)
    y = 0
    for s in range(d):
        dh, dw, _ = src[s].shape
        x = 0
        for i in range(w_s):
            dst[y:y + dh, x:x + dw] = \
                ((src[s][:, :, i] - np.min(src[s][:, :, i])) / np.max(src[s][:, :, i] - np.min(src[s][:, :, i])) * 255 + 0.5).astype(np.uint8)
            x += dw
        y += dh
    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gaussian_pyramids, DoG = DoG_pyramids(gray)

    DoG_result = show_pyramids(DoG, d=4, h_s=2, w_s=4 , name = 'DoG')
    gaussian_result = show_pyramids(gaussian_pyramids, d=4, h_s=2, w_s=5, name = 'Gau')

    cv2.imshow('DoG', DoG_result)
    cv2.imshow('Gaussian Pyramids', gaussian_result)
    cv2.waitKey()
    cv2.destroyAllWindows()
