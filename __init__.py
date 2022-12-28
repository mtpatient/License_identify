from tkinter import *

# 创建GUI窗口
root = Tk()
root.title("车牌识别")
root.geometry("530x330")

# 按钮
# 通过摄像头获取图片
btn_camera = Button(root, text="来自摄像头", height=1, width=10)
btn_camera.place(x=100, y=0)

# 上传图片
btn_img = Button(root, text="上传图片", height=1, width=10)
btn_img.place(x=1, y=0)

# 进入事件循环
root.mainloop()
