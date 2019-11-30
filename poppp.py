/*
this algorithm needs user select 3 imgs as target, the most K similar imgs and k connection
K represent the output of the most K similiar imgs
k_imgs represent the k connected imgs to the user pick target
*/
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import math


def InImgs(n1,n2,n3):
    f = os.listdir("./imgs")
    l = 0
    for i in f:
        l += 1

    output = np.zeros((1, l))

    count = 0
    for i in f:
        if i == n1 or i == n2 or i == n3:
            output[0][count] = 1/3
            count += 1
        else:
            count += 1

    return output,l


def PPR(k_imgs, target):
    f = os.listdir("./imgs")
    ui = 0
    for j in f:
        ui+=1
        print(j)

    numImgs = ui

    win_h, win_w = 100, 100
    colln_lbp = []
    for i in f:
        img = plt.imread("./imgs/" + i)
        img = rgb2gray(img)
        img_h, img_w = img.shape[0], img.shape[1]
        lbp = []
        for h in range(0, img_h - win_h + 1, win_h):
            for w in range(0, img_w - win_w + 1, win_w):
                win = img[h:h + win_h, w:w + win_w]
                (hist, _) = np.histogram((local_binary_pattern(win, 24, 8)).ravel(), bins=26, range=(0, 26))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                lbp.append(hist)
        colln_lbp.append(lbp)
    colln_lbp = np.array(colln_lbp)
    x, y, z = colln_lbp.shape
    colln_lbp = colln_lbp.reshape((x, y * z))

    out = np.zeros((numImgs, numImgs))

    for i in range(numImgs):
        for j in range(numImgs):
            if i == j:
                out[i][j] = 0
            else:
                out[i][j] = round(1 / round(np.sqrt(np.sum(np.square(colln_lbp[i][:] - colln_lbp[j][:]))), 2), 2)

    sum1 = np.zeros((1, numImgs))
    ss = np.zeros((numImgs,numImgs))
    u = k_imgs
    co = 0
    while u>0:
        for ww in range(numImgs):
            max = out[0][ww]
            for www in range(numImgs):
                if out[www][ww]>max:
                    max = out[www][ww]
                    co = www
            ss[co][ww] = max
            out[co][ww] = 0
            co = 0
        u = u - 1


    for j in range(numImgs):
        for i in range(numImgs):
            sum1[0][j] = sum1[0][j] + ss[i][j] #

    out1 = np.zeros((numImgs, numImgs))

    for i in range(numImgs):
        for j in range(numImgs):
            out1[j][i] = round(ss[j][i] / sum1[0][i], 2)

    #print(out)

    sum2 = np.zeros((1, numImgs))
    for j in range(numImgs):
        for i in range(numImgs):
            sum2[0][j] = sum2[0][j] + out1[i][j]
        if sum2[0][j] - 1 > 0:
            print(sum2[0][j])
            out1[i][j] = out1[i][j] - sum2[0][j] + 1
            print("wrong here")
        if sum2[0][j] - 1 < 0:
            print(sum2[0][j])
            out1[i][j] = out1[i][j] + sum2[0][j] - 1
            print("wrong")

    print(sum1)

    print(out1)

    b = np.array([[1/2, 0, 0, 0, 0, 1/2]])
    b = b.transpose()
    print(b)

    target = target.transpose()

    for i in range(30):
        target = out1.dot(target)

    print(target)

    return target


def most_K(target_after, l, K):
    target_after = target_after.transpose()
    out = np.zeros((1,K))
    for i in range(K):
        max = target_after[0][0]
        count = 0
        for j in range(l):
            if target_after[0][j] >= max:
                max = target_after[0][j]
                count = j
        target_after[0][count] = 0
        out[0][i] = count

    return out


if __name__ == "__main__":
    K = 4
    k_img = 5
    l = 0
    img_list, l = InImgs("Hand_0000012.jpg", "Hand_0000013.jpg", "Hand_0000040.jpg")

    target = img_list

    target_after = PPR(k_img, target)

    out = most_K(target_after, l, K)

    ff = os.listdir("./imgs")

    for iii in range(K):
        print(ff[int(out[0][iii])])

