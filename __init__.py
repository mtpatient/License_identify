from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import cv2
import function
import time


# 导入图片 图片路径不能包含中文
def loadImg():
    file_path = filedialog.askopenfilename(title=u'选择图片')
    # print("图片路径：" + file_path)

    img_open = Image.open(file_path)
    global img  # 全局变量声明
    img = ImageTk.PhotoImage(img_open)

    Label_img["image"] = img
    car_img = cv2.imread(file_path)

    time_start = time.time()

    global carLicense, text  # 全局变量声明
    carLicense, text = function.getRes(car_img)

    # plt.imshow(carLicense)
    # plt.show()

    carLicense = Image.fromarray(carLicense)
    carLicense = ImageTk.PhotoImage(carLicense)

    label0['image'] = carLicense
    label0.pack()

    label1['text'] = text
    label1.pack()

    time_end = time.time()

    label2['text'] = str(time_end - time_start) + '  秒'
    label2.pack()


# 创建GUI窗口
root = Tk()
root.title("车牌识别")
root.geometry("800x500")

# 左区域
left_frame = Frame(root, height=500, width=460)
left_frame.pack(side='left')

# 上传图片按钮
btn_img = Button(left_frame, text="导入图片", height=1, width=10, command=lambda: loadImg())
btn_img.place(x=0, y=0)

Label_img = Label(left_frame, bg='Gainsboro', height=420, width=460)
Label_img.place(x=0, y=32)

# 右区域
right_frame = Frame(root, height=500, width=300)
right_frame.pack(side='right')

Label_res_title = Label(right_frame, text='识别结果:', font=("微软雅黑", 12))
Label_res_title.place(x=0, y=0)

# 显示车牌
Label_res_img_text = Label(right_frame, text='车牌位置：', font=("微软雅黑", 10))
Label_res_img_text.place(x=1, y=30)

Frame_res_img = Frame(right_frame, height=80, width=300)
Frame_res_img.place(x=0, y=54)

label0 = Label(Frame_res_img, height=40, width=300, bg='Gainsboro')

# 识别结果
Label_res_text_position = Label(right_frame, text='识别结果：', font=("微软雅黑", 10))
Label_res_text_position.place(x=1, y=135)

Frame_res_text = Frame(right_frame, height=80, width=300)
Frame_res_text.place(x=0, y=164)

label1 = Label(Frame_res_text, font=("微软雅黑", 16), justify='center', padx=100, bg='Gainsboro')

# 运行时间
Label_time = Label(right_frame, text='运行时间:', font=("微软雅黑", 10))
Label_time.place(x=1, y=260)

Frame_time_text = Frame(right_frame, height=40, width=300)
Frame_time_text.place(x=0, y=290)

label2 = Label(Frame_time_text, font=("微软雅黑", 16), justify='left', bg='Gainsboro')

# 进入事件循环
root.mainloop()
