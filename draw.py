from tkinter import *
from tkinter import ttk, colorchooser
from PIL import Image, ImageDraw
import numpy as np
from numpy import asarray
import json
import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json
import os

# Do not use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class main:
    def __init__(self,master):
        self.master = master
        self.color_fg = 'white'
        self.color_bg = 'black'
        self.color_middle = 'gray'
        self.old_x = None
        self.old_y = None
        self.penwidth = 40

        #MNIST parameters
        self.mean=33.791224489795916
        self.std=79.17246322228644

        self.guess = None
        self.count=0
        self.count_max =10

        self.image=Image.new(mode = "L", size = (28,28), color=0)
        self.loadNN() 

        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)#drawing the line 
        self.c.bind('<B3-Motion>',self.erase)#erasing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)

        self.prepCanvas()

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)
            draw=ImageDraw.Draw(self.image)
            draw.line((self.old_x/20,self.old_y/20,e.x/20,e.y/20),fill=self.color_fg,width = 2)

        if self.count == self.count_max:
            self.updateScores()
            self.count = 0
        self.count += 1

        self.old_x = e.x
        self.old_y = e.y

    def updateScores(self):
        if  self.old_x and self.old_y:
            data =  asarray(self.image)
            data = data.reshape(data.shape[0],data.shape[1],1)
            data = ((data - self.mean) / self.std).astype('float32')

            self.cProb.delete(ALL)
            self.prepCanvas()

            probabilities = self.NN.predict(np.array([data]))
            for i in range(10):
                self.cProb.create_line(50+0,30+30*i,50+round(100*probabilities[0][i]),30+30*i,width=20,fill='red')      


    def erase(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth+20,fill=self.color_bg,capstyle=ROUND,smooth=True)
            draw=ImageDraw.Draw(self.image)
            draw.line((self.old_x/20,self.old_y/20,e.x/20,e.y/20),fill=self.color_bg,width = 3)

        self.updateScores()

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):    
        self.old_x = None
        self.old_y = None      

    def clear(self):
        self.c.delete(ALL)
        self.image=Image.new(mode = "L", size = (28,28), color=0)
        self.cProb.delete(ALL)
        self.prepCanvas()


    def loadNN(self):
        json_file = open('model_files\\model.json', 'r')
        loaded_model_json = json_file.read()

        model = model_from_json(loaded_model_json)
        model.load_weights('model_files\\model.h5')

        json_file.close()
        self.NN = model

    def guessNumber(self):
        self.image.save('number.png') 
        data =  asarray(self.image)
        data = data.reshape(data.shape[0],data.shape[1],1)
        data = ((data - self.mean) / self.std).astype('float32')

        #self.guess=self.NN.predict_classes(np.array([data]))[0]
        self.guess=np.argmax(self.NN.predict(np.array([data])), axis=-1)[0]
        self.label.config(text = 'Number: '+ str(self.guess))

    def drawWidgets(self):
        self.controls = Frame(self.master, padx = 5, pady = 5)

        self.button = ttk.Button(self.controls,text = 'Save Number',command=self.guessNumber)
        self.button.grid(row=0,column=1,ipadx=30)
        self.controls.pack(side=LEFT)

        self.label = Label(self.controls, text = 'Number: '+ str(self.guess), font=('Helvetica', 18))
        self.label.grid(row=1, column=1)
        self.controls.pack(side=LEFT)

        self.buttonDEL = ttk.Button(self.controls, text = 'Clear', command = self.clear)
        self.buttonDEL.grid(row=2,column=1,ipadx=30)
        self.controls.pack(side=LEFT)

        self.cProb = Canvas(self.controls,width=100,height=400,bg='white',)
        self.cProb.grid(row=3,column=1,ipadx=30)
        self.controls.pack(side=LEFT)

        self.c = Canvas(self.master,width=28*20,height=28*20,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=False)

    def prepCanvas(self):
        for i in range(10):
            self.cProb.create_text(10,30+30*i,text=str(i))

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Number Guesser')
    root.resizable(False, False)
    root.mainloop()