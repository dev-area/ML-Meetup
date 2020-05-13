from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageGrab,Image,ImageDraw
import numpy as np
import pandas as pd
import sklearn.neural_network as sknet
import sklearn.model_selection as skmodel
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


model = None

class Paint(object):

    DEFAULT_PEN_SIZE = 15.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.check_button = Button(self.root, text='Check', command=self.check_digit)
        self.check_button.grid(row=0, column=0)

        self.clear_button = Button(self.root, text='Clear', command=self.clear_canvas)
        self.clear_button.grid(row=0, column=1)

        self.tres = Text(self.root, height=3, width=30)
        self.tres.grid(row=0,column=2)
        self.tres.configure(font=("Times New Roman", 22, "bold"))

        self.bltext = "Learn"
        self.learn_button = Button(self.root, text=self.bltext, command=self.learn_new)
        self.learn_button.grid(row=0, column=3)


        self.c = Canvas(self.root, bg='White', width=600, height=600)
        self.c.grid(row=1, columnspan=5)

        self.image1 = Image.new("L", (600, 600))
        self.draw = ImageDraw.Draw(self.image1)


        self.setup()
        self.root.mainloop()

    def clear_canvas(self):
        self.c.delete('all')
        self.image1 = Image.new("L", (600, 600))
        self.draw = ImageDraw.Draw(self.image1)
        self.learn_button['text'] = "Learn"

    def check_digit(self):
        global model
        self.image1.thumbnail((28,28))
        self.image1.save("d1.bmp")
        self.arr = np.array(self.image1)
        r = model.predict(self.arr.reshape(1,784).reshape(1, 28, 28, 1))
        self.tres.delete(1.0, "end")
        self.tres.insert("end", r[0].argmax())

    def learn_new(self):
        v = int(self.tres.get('1.0', 'end'))
        model.partial_fit(pd.DataFrame(self.arr.reshape(1,784)),pd.Series(np.array([v])))
        self.learn_button['text'] = "Done"

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)


    def paint(self, event):
        self.line_width = 35
        paint_color = 'black'
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y],255,30)
        self.old_x = event.x
        self.old_y = event.y


    def reset(self, event):
        self.old_x, self.old_y = None, None


def train():
    global model
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.load_weights("sw.dat")

if __name__ == '__main__':
    train()
    print("ready")
    Paint()

