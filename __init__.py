from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


# 导入图片
def load_img():
    file_path = filedialog.askopenfilename(title=u'选择图片')
    print("图片路径：")
    print(file_path)

    img_open = Image.open(file_path)
    img = ImageTk.PhotoImage(img_open)

    Label_img.image = img


# 创建GUI窗口
root = Tk()
root.title("车牌识别")
root.geometry("800x500")

# 左区域
left_frame = Frame(root, height=500, width=460)
left_frame.pack(side='left')

# 上传图片按钮
btn_img = Button(left_frame, text="上传图片", height=1, width=10, command=lambda: load_img())
btn_img.place(x=0, y=0)

Label_img = Label(left_frame, bg='Gainsboro', height=420, width=460)
Label_img.place(x=0, y=32)

# 右区域
right_frame = Frame(root, height=500, width=300, bg='cyan')
right_frame.pack(side='right')

Label_res_title = Label(right_frame, text='识别结果:', font=("微软雅黑", 12))
Label_res_title.place(x=0, y=0)

# 进入事件循环
root.mainloop()
