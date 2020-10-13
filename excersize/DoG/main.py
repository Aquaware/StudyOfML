# -*- coding: utf-8 -*-
# OpenCV　XDoGによる輪郭抽出
# see. https://qiita.com/Shirataki2/items/813fdade850cc69d1882
#
import cv2
import numpy as np

def DoGFilter(image, kernel_size, sigma, k, gamma):
    g1 = cv2.GaussianBlur(image, kernel_size, sigma)
    g2 = cv2.GaussianBlur(image, kernel_size, sigma * k )
    return g1 - g2 * gamma

# 閾値で白黒化するDoG
def thres_dog(img, size, sigma, eps, k=1.6, gamma=0.98):
    d = DoG(img,size, sigma, k, gamma)
    d /= d.max()
    d *= 255
    d = np.where(d >= eps, 255, 0)
    return d

# 拡張ガウシアン差分フィルタリング
def xdog(img, size, sigma, eps, phi, k=1.6, gamma=0.98):
    eps /= 255
    d = DoG(img,size, sigma, k, gamma)
    d /= d.max()
    e = 1 + np.tanh(phi*(d-eps))
    e[e>=1] = 1
    return e

# シャープネス値pを使う方
def pxdog(img, size, p, sigma, eps, phi, k=1.6):
    eps /= 255
    g1 = cv2.GaussianBlur(img, (size, size), sigma)
    g2 = cv2.GaussianBlur(img, (size, size), sigma*k)
    d = (1 + p) * g1 - p * g2
    d /= d.max()
    e = 1 + np.tanh(phi*(d-eps))
    e[e>=1] = 1
    return e

def test():
    img = cv2.imread('./image.jpg')
    height, width, channels = img.shape[:3]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img2 = DoGFilter(gray, (7, 7), 1.1, 2.2, 1)

    cv2.imshow('window1', gray)
    cv2.imshow('window2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test()