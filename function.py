import cv2
import numpy as np


# 实现图片的定位
def getPostion(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = img_gray.copy()
    # 高斯滤波 将噪声
    gray_img = cv2.GaussianBlur(img_copy, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    # 开运算 将图像中不同部分分开
    kernel = np.ones((23, 23), np.uint8)
    img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0)  # 开运算后的图片与原始的灰度图片按一定权重进行一个融合
    # 找到图像边缘
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 阈值分割
    img_edge = cv2.Canny(img_thresh, 100, 200)  # canny 边缘检测
    # # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((9, 9), np.uint8)
    img_edge2 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    # img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel) # 我测试时发现有这句代码，开运算就会把中文那部分去除
    # # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 从找到的轮廓中找出车牌的位置
    temp_contours = []
    areaAverage = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # print("面积 ", area)
        areaAverage.append(area)
    minArea = np.mean(areaAverage)
    # print("平均面积", minArea)  #我们假设车牌的面积一定大于平均面积
    temp_contours = []  # 先筛除较小的轮廓
    for contour in contours:
        if cv2.contourArea(contour) > minArea:
            temp_contours.append(contour)
    car_plates = []  # 可能的车牌
    for temp_contour in temp_contours:
        rect = cv2.minAreaRect(temp_contour)  # 该函数返回一个rect对象， rect[0]--中心点  rect[1] -- 矩形的长、宽
        rect_width, rect_height = rect[1]
        if rect_width < rect_height:  # 调整长宽
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if 2 < aspect_ratio < 5.5:
            car_plates.append(temp_contour)
    # 一车只有一个车牌
    if len(car_plates) == 1:
        for car_plate in car_plates:
            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)  # 左上角的坐标
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)  # 右下角的坐标
            # cv2.rectangle(img, (row_min, col_min), (row_max, col_max), (0, 255, 0), 2) # 在原图中画矩形框
            card_img = img[col_min:col_max, row_min:row_max, :]  # 把矩形框中的图像显示出来
            return card_img
    else:
        print("查找车牌失败")
        return None
