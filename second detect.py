import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def detectt(img):
    im = cv2.imread(img, cv2.IMREAD_COLOR)
    #gray = cv2.fastNlMeansDenoisingColored(im, None, 10, 3, 3, 3)#通过fastNlMeansDenoising对图像进行去噪
    #coefficients = [0, 1, 1]
    #m = np.array(coefficients).reshape((1, 3))
    #gray = cv2.transform(gray, m)#将图片转灰色
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊处理
    retval, gray = cv2.threshold(gray, 80, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)  # otsu二值化操作
    cv2.imshow("gray2",gray)
    cannyimg = cv2.Canny(gray, gray.shape[0], gray.shape[1])  # 对二值化的原图进行canny边缘检测
    cv2.imshow("cann",cannyimg)

    yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)  # RGB转YUV
    # cv2.imshow("yuv", yuv)

    hi = yuv.shape[0]
    wi = yuv.shape[1]
    (Y, U, V) = cv2.split(yuv)  # 通道拆分，为之后的通道合并做准备
    # cv2.imshow("U",U)

    cannyV = cv2.Canny(V, 50, 150)
    cv2.imshow("cannyV",cannyV)
    zero = np.zeros((hi, wi), np.uint8)
    newuv = cv2.merge([zero, U, V])  # 合并U,V通道
    # cv2.imshow("newuv",newuv)
    cannyNewuv = cv2.Canny(newuv, 50, 150)  # 通过canny边缘检测得出图片背景花纹
    cv2.imshow("cannyNewuv",cannyNewuv)

    cannyU = cv2.Canny(U, 50, 150)
    cv2.imshow("cannyU",cannyU)

    height = cannyimg.shape[0]
    width = cannyimg.shape[1]
    canny2 = cv2.Canny(yuv, 50, 150)
    tiger1 = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            if ((cannyimg[i, j] == cannyNewuv[i, j]) & (cannyNewuv[i, j] == 255)):  # 消除背景花纹
                color = 0
            else:
                color = cannyimg[i, j]
            tiger1[i, j] = np.uint8(color)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    z = [0] * height
    v = [0] * width
    print('shape for :',cannyimg.shape)
    hfg = [[0 for col in range(2)] for row in range(height)]
    lfg = [[0 for col in range(2)] for row in range(width)]
    box = [0, 0, 0, 0]
    # 水平投影
    a = 0
    q = 0
    emptyImage1 = np.zeros((height, width, 3), np.uint8)

    for y in range(0, height):
        for x in range(0, width):
            cp = tiger1[y, x]
            # if np.any(closed[y,x]):
            if cp == 255:
                a = a + 1
            else:
                continue
        z[y] = a
        q = q + a
        # print z[y]
        a = 0

    bq = 3 * q / height
    # 根据水平投影值选定行分割点
    inline = 1
    start = 0
    j = 0
    print(z)
    for i in range(0, height):
        if inline == 1 and z[i] >= 30:  # 从空白区进入文字区
            start = i  # 记录起始行分割点
            # print i
            inline = 0
            print('nani')
        elif (i - start > 5) and z[i] < 30 and inline == 0:  # 从文字区进入空白区
            inline = 1
            hfg[j][0] = start - 2  # 保存行分割位置
            hfg[j][1] = i + 2
            j = j + 1
            print('nani')
            break
    # 对每一行垂直投影、分割
    a = 0
    c = 0

    for p in range(0, j):
        for x in range(0, width):
            for y in range(hfg[p][0], hfg[p][1]):
                print('kan:', hfg[p][0],p)
                cp1 = tiger1[y, x]
                if cp1 == 255:
                    a = a + 1
                else:
                    continue
            v[x] = a  # 保存每一列像素值
            c = c + a
            a = 0
        qc = c / (3 * width)
        # print width
        # 垂直分割点
        incol = 1
        start1 = 0
        j1 = 0
        z1 = hfg[p][0]
        z2 = hfg[p][1]
        h1 = 0
        h2 = 0
        xnum = 0
        for i1 in range(0, width):
            if incol == 1 and v[i1] >= 2:  # 从空白区进入文字区
                start1 = i1  # 记录起始列分割点
                incol = 0
            elif (i1 - start1 > 13) and v[i1] < 8 and incol == 0:  # 从文字区进入空白区
                incol = 1
                lfg[j1][0] = start1 - 2  # 保存列分割位置
                lfg[j1][1] = i1 + 2
                print(i1)
                if (start - 2) <= 0:
                    l1 = start
                    print(start + "why")
                else:
                    l1 = start1
                    print("hehe" + str(start))
                h1 = l1
                l2 = i1
                h2 = l2
                j1 = j1 + 1
                xnum += 1
            ximg = im[z1:z2, h1:h2]
            image_save_path_head = "F:/banksave/image/xnum"
            image_save_path_tail = ".jpg"
            image_save_path = "%s%d%s" % (image_save_path_head, xnum, image_save_path_tail)  ##将整数和字符串连接在一起

            cv2.imwrite(image_save_path, ximg)
            # cv2.imwrite('F:/banksave/image/ximg12.png', ximg)
            #cv2.rectangle(img, (h1, z1), (h2, z2), (255, 0, 0), 2)
    # cv2.imshow('final', img)
    # cv2.imwrite('F:/banksave/image/final.png', img)
    # tiger1 = cv2.morphologyEx(tiger1, cv2.MORPH_CLOSE, kernel)
    # tiger1 = cv2.dilate(tiger1,kernel,iterations=1)
    #tiger1 = cv2.erode(tiger1, kernel, iterations=1)
    tiger1 = cv2.morphologyEx(tiger1, cv2.MORPH_CLOSE, kernel)  # 闭操作除去较小黑点，填充连通的白色区域

    cv2.imshow("tiger1", tiger1)

    hx= im.shape[0]
    wx = im.shape[1]
    six = wx //4
    image_save_path_head = "F:/banksave/image/xx"
    image_save_path_tail = ".jpg"
    seq = 1
      # [1]480*360==15*11---height
    for j in range(4):  # [2]column-----------width
        if (j*six-20)<0:
            img_roi = im[0:hx, (j * six):((j + 1) * six-20)]
        else:
            img_roi = im[0:hx, (j * six)-20:((j + 1) * six - 20)]
        image_save_path = "%s%d%s" % (image_save_path_head, seq, image_save_path_tail)  ##将整数和字符串连接在一起
        #cv2.imwrite(image_save_path, img_roi)
        seq = seq + 1

        # 通过水平，垂直投影定位分割卡号


detectt("F:/banksave/image/ximg12.png")
cv2.waitKey(0)