import cv2
import numpy as np


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


# 预处理图像，并返回车牌位置
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
