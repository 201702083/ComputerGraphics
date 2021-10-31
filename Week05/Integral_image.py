import numpy as np

def get_integral_image(src):
    assert len(src.shape) == 2
    h, w = src.shape
    dst = np.zeros(src.shape)
    for row in range(h):
        dst[row,0] = np.sum(src[0:row+1, 0])
    for col in range(w):
        dst[0,col] = np.sum(src[0,0:col+1])
    for row in range(1,h):
        for col in range(1,w):
            dst[row,col] = src[row,col] + dst[row-1,col] + dst[row,col-1] - dst[row-1, col-1]

    return dst

def calc_local_integral_value(src, left_top, right_bottom):
    assert len(left_top) == 2
    assert len(right_bottom) == 2

    lt_row , lt_col = left_top
    rb_row , rb_col = right_bottom

    lt_val = src[lt_row -1, lt_col -1]
    rt_val = src[rb_row -1, rb_col]
    lb_val = src[rb_row, lt_col-1]
    rb_val = src[rb_row, rb_col]

    if lt_row == 0 :
        lt_val = 0
        lb_val = 0

    if lt_col == 0 :
        lt_val = 0
        rt_val = 0

    return lt_val - lb_val - rt_val + rb_val

def main():
    src = np.array([[31, 2, 4, 33, 5, 36],
           [12, 26, 9, 10, 29, 25],
           [13, 17, 21, 22, 20, 18],
           [24, 23, 15, 16, 14, 19],
           [30, 8, 28, 27, 11, 7],
           [1, 35, 34, 3, 32, 6]])

    integral_src = get_integral_image(src)
    row, col = 2, 3
    b = 3

    sum = 0
    ## 22 + 20 + 18 + 16 + 14 + 19 + 27 + 11 + 7
    for i in range(row, row+b):
        for j in range(col, col+b):
            sum += src[i, j]

    ## 84 + 555 - 222 - 263
    integral_sum = calc_local_integral_value(integral_src, (row, col), (row+b-1, col+b-1))

    print("image: \n{}".format(src))
    print("integral image: \n{}".format(integral_src))

    print("sum [{}:{}, {}:{}]".format(row, row+b, col, col+b))
    print("image: {}".format(sum))
    print("integral image: {}".format(integral_sum))
    return


if __name__ == '__main__':
    main()