import math
from msilib import Table
from tkinter import *



def button_click():
    S1 = "00"
    S2 = "01"
    S3 = "10"
    S4 = "11"




    print("X1X2 Q1Q2 q1q2 Y1Y2 ")

    S13x = str(bin(int(x1.get()))[2:])
    S12x = str(bin(int(x2.get()))[2:])
    S32x = str(bin(int(x3.get()))[2:])
    S23x = str(bin(int(x4.get()))[2:])
    S34x = str(bin(int(x5.get()))[2:])
    S41x = str(bin(int(x6.get()))[2:])
    S44x = str(bin(int(x7.get()))[2:])

    S13y = str(bin(int(y1.get()))[2:])
    S12y = str(bin(int(y2.get()))[2:])
    S32y = str(bin(int(y3.get()))[2:])
    S23y = str(bin(int(y4.get()))[2:])
    S34y = str(bin(int(y5.get()))[2:])
    S41y = str(bin(int(y6.get()))[2:])
    S44y = str(bin(int(y7.get()))[2:])

    x = [S13x, S12x, S32x, S23x, S34x, S41x, S44x]
    y = [S13y, S12y, S32y, S23y, S34y, S41y, S44y]

    for i in range(0, 7):
        if len(x[i]) == 1:
            x[i] = "0" + x[i]

    for j in range(0, 7):
        if len(y[j]) == 1:
            y[j] = "0" + y[j]




    d = [int(S13x, 2), int(S12x, 2), int(S32x, 2), int(S23x, 2), int(S34x, 2), int(S41x, 2), int(S44x, 2)]

    r = [" " + x[0] + "   " + S1 + "   " + S3 + "   " + y[0],
         " " + x[1] + "   " + S1 + "   " + S2 + "   " + y[1],
         " " + x[2] + "   " + S3 + "   " + S2 + "   " + y[2],
         " " + x[3] + "   " + S2 + "   " + S3 + "   " + y[3],
         " " + x[4] + "   " + S3 + "   " + S4 + "   " + y[4],
         " " + x[5] + "   " + S4 + "   " + S1 + "   " + y[5],
         " " + x[6] + "   " + S4 + "   " + S4 + "   " + y[6]

         ]



    n = len(d)

    for i in range(n):
        for j in range(0, n - i - 1):
            if d[j] > d[j + 1]:
                d[j], d[j + 1] = d[j + 1], d[j]
                r[j], r[j + 1] = r[j + 1], r[j]

    for i in r:
        print(i)









root = Tk()

root['bg'] = '#fafafa'
root.title('1 Работа')
root.geometry('1920x1080')




# Создаем холст
canvas = Canvas(root, width=1000, height=1000, background='white')
canvas.pack()
# Координаты и радиус круга

radius = 80
x = 500
y = 100

canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline="black")
canvas.create_text(x, y, text="S1", font=("Arial", 12, "bold"))


canvas.create_oval(x + 200 - radius, y + 200 - radius, x + 200 + radius, y + 200 + radius, outline="black")
canvas.create_text(x + 200, y + 200, text="S2", font=("Arial", 12, "bold"))

canvas.create_oval(x - 200 - radius, y + 200 - radius, x - 200 + radius, y + 200 + radius, outline="black")
canvas.create_text(x - 200, y + 200, text="S3", font=("Arial", 12, "bold"))

canvas.create_oval(x - radius, y + 400 - radius, x + radius, y + 400 + radius, outline="black")
canvas.create_text(x, y + 400, text="S4", font=("Arial", 12, "bold"))

# Рисуем линию, представляющую тело стрелки
canvas.create_line(x - radius, y, x - 120 - radius, y + 200 - radius, width=2, arrow=LAST)

canvas.create_line(x + radius, y, x + 120 + radius, y + 200 - radius, width=2, arrow=LAST)

canvas.create_line(x - 200 + radius, y + 180, x + 200 - radius, y + 180, width=2, arrow=LAST)
canvas.create_line(x + 200 - radius, y + 220, x - 200 + radius, y + 220, width=2, arrow=LAST)

canvas.create_line(x - 120 - radius, y + 200 + radius, x - radius, y + 400, width=2, arrow=LAST)

canvas.create_line(x, y + 400 - radius, x, y + radius, width=2, arrow=LAST)


line_id = canvas.create_line(x - 20, y + 400 + radius, x, y + 500 + radius, x + 20, y + 400 + radius, smooth=True, width=2, fill="black", arrow=LAST)




canvas.create_text(330, 100, text="x", font=("Arial", 12, "bold"))
x1 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window8 = canvas.create_window(350, 100, window=x1, anchor=CENTER)

canvas.create_text(370, 100, text="y", font=("Arial", 12, "bold"))
y1 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window9 = canvas.create_window(390, 100, window=y1, anchor=CENTER)



canvas.create_text(630, 100, text="x", font=("Arial", 12, "bold"))
x2 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window10 = canvas.create_window(650, 100, window=x2, anchor=CENTER)

canvas.create_text(670, 100, text="y", font=("Arial", 12, "bold"))
y2 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window11 = canvas.create_window(690, 100, window=y2, anchor=CENTER)





canvas.create_text(380, 260, text="x", font=("Arial", 12, "bold"))
x3 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window = canvas.create_window(400, 260, window=x3, anchor=CENTER)

canvas.create_text(420, 260, text="y", font=("Arial", 12, "bold"))
y3 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window1 = canvas.create_window(440, 260, window=y3, anchor=CENTER)



canvas.create_text(540, 300, text="x", font=("Arial", 12, "bold"))
x4 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window2 = canvas.create_window(560, 300, window=x4, anchor=CENTER)

canvas.create_text(580, 300, text="y", font=("Arial", 12, "bold"))
y4 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window3 = canvas.create_window(600, 300, window=y4, anchor=CENTER)


canvas.create_text(230, 400, text="x", font=("Arial", 12, "bold"))
x5 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window12 = canvas.create_window(250, 400, window=x5, anchor=CENTER)

canvas.create_text(270, 400, text="y", font=("Arial", 12, "bold"))
y5 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window13 = canvas.create_window(290, 400, window=y5, anchor=CENTER)




canvas.create_text(510, 400, text="x", font=("Arial", 12, "bold"))
x6 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window4 = canvas.create_window(530, 400, window=x6, anchor=CENTER)

canvas.create_text(550, 400, text="y", font=("Arial", 12, "bold"))
y6 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window5 = canvas.create_window(570, 400, window=y6, anchor=CENTER)



canvas.create_text(510, 650, text="x", font=("Arial", 12, "bold"))
x7 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window6 = canvas.create_window(530, 650, window=x7, anchor=CENTER)

canvas.create_text(550, 650, text="y", font=("Arial", 12, "bold"))
y7 = Entry(root, width=2, bd=2, relief=SOLID)
entry_window7 = canvas.create_window(570, 650, window=y7, anchor=CENTER)










button = Button(root, text="Start", command=button_click)
button_window = canvas.create_window(400, 700, window=button, anchor=CENTER)




root.mainloop()
