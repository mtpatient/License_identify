import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from cnocr import CnOcr
import re


# 显示彩色图片
def plt_show_rgb(img):
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# 显示灰度图
def plt_show_gray(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# 拉伸图像
def stretch(img):
    Max = float(img.max())
    Min = float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255 / (Max - Min) * img[i, j] - (255 * Min) / (Max - Min))

    return img


# 图像二值化
def toBinarization(img):
    Max = float(img.max())
    Min = float(img.min())

    x = Max - ((Max - Min) / 2)
    # 二值化,返回阈值ret  和  二值化操作后的图像thresh
    ret, thresh = cv2.threshold(img, x, 255, cv2.THRESH_BINARY)
    # 返回二值化后的黑白图像
    return thresh


# 获取轮廓左上角和右下角的坐标
def getRect(contour):
    y, x = [], []

    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])

    return [min(y), min(x), max(y), max(x)]


# 定位车牌位置
def locateLicense(img, img0):
    # 画出轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找出最大的三个区域
    block = []
    for c in contours:
        # 计算面积和长度比，
        r = getRect(c)
        s = (r[2] - r[0]) * (r[3] - r[1])  # 面积
        l = (r[2] - r[0]) * (r[3] - r[1])  # 长度比

        block.append([r, s, l])
    # 选出面积最大的3个区域
    block = sorted(block, key=lambda b: b[1])[-3:]

    # 颜色识别
    maxWeight, maxIndex = 0, -1
    for i in range(len(block)):
        b = img0[block[i][0][1]:block[i][0][3], block[i][0][0]:block[i][0][2]]

        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        # 车牌的范围
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        # 根据阈值构建掩膜
        mask = cv2.inRange(hsv, lower, upper)
        # 统计权值
        w1 = 0
        for m in mask:
            w1 += m / 255

        w2 = 0
        for n in w1:
            w2 += n

        # 选出最大权值的区域
        if w2 > maxWeight:
            maxIndex = i
            maxWeight = w2

    return block[maxIndex][0]


# 预处理图像，并根据预处理好的图像返回车牌位置
def getPosition(img):
    # 压缩图像
    m = 400 * img.shape[0] / img.shape[1]
    img = cv2.resize(img, (400, int(m)), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_stretch = stretch(img_gray)  # 拉伸图像

    # 开运算
    r = 16
    h = w = r * 2 + 1
    kernel = np.zeros((h, w), np.uint8)
    cv2.circle(kernel, (r, r), r, 1, -1)
    img_open = cv2.morphologyEx(img_stretch, cv2.MORPH_OPEN, kernel)

    img_temp = cv2.absdiff(img_stretch, img_open)

    # 图像二值化
    img_binary = toBinarization(img_temp)

    # 边缘检测
    canny = cv2.Canny(img_binary, img_binary.shape[0], img_binary.shape[1])

    # 闭运算
    kernel = np.ones((5, 19), np.uint8)
    img_close = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

    # 开运算
    img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11, 5), np.uint8)
    img_open = cv2.morphologyEx(img_open, cv2.MORPH_OPEN, kernel)

    # 定位车牌位置，获取矩形
    rect = locateLicense(img_open, img)

    return rect, img


# 识别车牌
def getText(img):
    ocr = CnOcr()  # 获取ocr对象
    # t = ocr.ocr_for_single_line(img)
    # print(t)
    image = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯去噪
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # plt_show_gray(gray_image)

    # 自适应阈值处理
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    # image = toBinarization(image)
    image = cv2.bitwise_not(image)

    kernel = np.ones((1, 1), dtype=np.uint8)
    # image = cv2.erode(image, kernel, iterations=1)  # 腐蚀
    image = cv2.dilate(image, kernel, iterations=1)  # 膨胀

    # plt_show_gray(image)

    text = ocr.ocr_for_single_line(image)
    # text = ocr.ocr(image)

    '''
    分割字符识别
    '''
    # words = splitLicense(img)
    # text = pytesseract.image_to_string(image, lang="chi_sim+eng", config="--psm 7")

    # text = []
    # for w in words:
    #     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #     # w0 = cv2.dilate(w, kernel)
    #     # plt_show_gray(w0)
    #     t = pytesseract.image_to_string(w,
    #                                     config=f'-l chi_sim')  # 识别文字
    #     text.append(t)

    # print(text)
    res = text['text'].upper()
    if res[2].isalpha():
        res = res[:2] + '-' + res[2:]
    else:
        res = res[:2] + '-' + res[3:]

    print(res)

    return res


# 分割字符
def splitLicense(img):
    image = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯去噪
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 自适应阈值处理
    ret, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU)
    img1 = image.copy()
    image2 = cv2.bitwise_not(image)
    # text = pytesseract.image_to_string(image2,
    #                                    config=f'--psm 8 chi_sim --oem 3 -c '
    #                                           f'tessedit_char_whitelist=鄂赣甘贵桂黑沪冀津京吉辽鲁蒙闽宁青琼陕苏晋皖湘新豫渝粤云藏浙'
    #                                           f'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    text = pytesseract.image_to_string(image2, lang="chi_sim+eng", config="--psm 6")
    print(text)
    plt_show_gray(image2)

    # 闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
    # 白色部分膨胀
    image = cv2.dilate(image, kernel)
    # plt_show_gray(image)

    # 轮廓检测
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img0 = img.copy()
    cv2.drawContours(img0, contours, -1, (0, 255, 0), 2)
    plt.imshow(img0)
    plt.show()

    # 筛选出各个字符位置的轮廓
    words = []
    for i in contours:
        word = []
        rect = cv2.boundingRect(i)
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        word.append(x)
        word.append(y)
        word.append(width)
        word.append(height)
        words.append(word)

    words = sorted(words, key=lambda s: s[0], reverse=False)

    res = []
    cnt = 0
    for w in words:
        if (w[3] > (w[2] > 1.8)) and (w[3] < (w[2] * 2.5)):
            cnt += 1
            image0 = img1[w[1]:w[1] + w[3], w[0]:w[0] + w[2]]
            image0 = cv2.bitwise_not(image0)
            res.append(image0)
            # plt_show_gray(image0)

    return res


# 调用其它方法识别车牌，返回识别结果
def getRes(img):
    # 获取图像位置，并压缩图像
    rect, img = getPosition(img)
    x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])

    # 截取车牌
    res_img = img[y1:y2, x1:x2]
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    # 获取文本
    res_text = getText(res_img)

    return res_img, res_text
