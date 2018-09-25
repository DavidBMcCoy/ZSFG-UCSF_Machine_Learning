# -------------------------------------------------------------------------------
# Name:        Object bounding box label tool
# Purpose:     Label object bboxes for ImageNet Detection data
# Author:      David McCoy
# Created:     09/20/2019

#
# -------------------------------------------------------------------------------
from __future__ import division
from Tkinter import *
import tkMessageBox
from PIL import Image, ImageTk, ImageEnhance
import os
import glob
import random

# colors for the bboxes
COLORS=['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE=256, 256


class LabelTool():
    def __init__(self, master):
        # set up the main frame
        self.parent=master
        self.parent.title("2D LabelTool")
        self.frame=Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width=FALSE, height=FALSE)

        # initialize global state
        self.imageDir=''
        self.imageList=[]
        self.egDir=''
        self.egList=[]
        self.outDir=''
        self.cur=0
        self.total=0
        self.category=0
        self.imagename=''
        self.labelfilename=''
        self.tkimg=None

        # initialize augmented images
        self.curr_img=None
        self.bright_img=None
        self.contrast_img=None
        self.sharp_img=None

        # initialize windowing parameters
        self.bright_value = None
        self.contrast_value = None
        self.sharp_value = None

        # zoom functionality
        self.scale=1.0

        # initialize mouse state
        self.STATE={}
        self.STATE['click']=0
        self.STATE['x'], self.STATE['y']=0, 0

        # reference to bbox
        self.bboxIdList=[]
        self.bboxId=None
        self.bboxList=[]
        self.hl=None
        self.vl=None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label=Label(self.frame, text="Image Dir:")
        self.label.grid(row=0, column=0, sticky=E)
        self.entry=Entry(self.frame)
        self.entry.grid(row=0, column=1, sticky=W + E)
        self.ldBtn=Button(self.frame, text="Load", command=self.loadDir)
        self.ldBtn.grid(row=0, column=2, sticky=W + E)

        # main panel for labeling
        self.mainPanel=Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.mainPanel.bind("<4>", self.mouseWheel)
        self.mainPanel.bind("<5>", self.mouseWheel)
        self.mainPanel.bind("<Button-3>", self.move_from)
        self.mainPanel.bind("<B3-Motion>", self.move_to)

        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.parent.bind("s", self.cancelBBox)
        self.parent.bind("a", self.prevImage)  # press 'a' to go backforward
        self.parent.bind("d", self.nextImage)  # press 'd' to go forward
        self.mainPanel.grid(row=1, column=1, rowspan=4, sticky=W + N)

        # showing bbox info & delete bbox
        self.lb1=Label(self.frame, text='Bounding boxes:')
        self.lb1.grid(row=1, column=2, sticky=W + N)

        self.listbox=Listbox(self.frame, width=22, height=12)
        self.listbox.grid(row=2, column=2, sticky=N)

        # delete bounding box
        self.btnDel=Button(self.frame, text='Delete', command=self.delBBox)
        self.btnDel.grid(row=3, column=2, sticky=W + E + N)

        # clear all bounding boxes
        self.btnClear=Button(self.frame, text='Clear All', command=self.clearBBox)
        self.btnClear.grid(row=4, column=2, sticky=W + E + N)

        # reset image
        self.btnReset=Button(self.frame, text='Reset Image', command=self.resetImage)
        self.btnReset.grid(row=6, column=2, sticky=W + E + N)

        # adjust brightness
        self.img_bright_label=Label(self.frame, text='Adjust Brightness')
        self.img_bright_label.grid(row=6, column=1, sticky=N)

        self.brgSc=Scale(self.frame, from_=0.0, to=5.0, orient=HORIZONTAL, command=self.adjBright, length=200, width=10,
                         sliderlength=15, tickinterval=0.1, resolution=0.01)
        self.brgSc.grid(row=7, column=1, sticky=N)

        # adjust contrast
        self.img_contrast_label=Label(self.frame, text='Adjust Contrast')
        self.img_contrast_label.grid(row=8, column=1, sticky=N)

        self.contSc=Scale(self.frame, from_=0.0, to=5.0, orient=HORIZONTAL, command=self.adjContrast, length=200,
                          width=10, sliderlength=15, tickinterval=0.1, resolution=0.01)
        self.contSc.grid(row=9, column=1, sticky=N)

        # sharpness slider
        self.img_sharp_label=Label(self.frame, text='Adjust Sharpness')
        self.img_sharp_label.grid(row=10, column=1, sticky=N)

        self.sharpSc=Scale(self.frame, from_=0.0, to=5.0, orient=HORIZONTAL, command=self.adjSharp, length=200,
                           width=10, sliderlength=15, tickinterval=0.1, resolution=0.01)
        self.sharpSc.grid(row=11, column=1, sticky=N)

        # control panel for image navigation
        self.ctrPanel=Frame(self.frame)
        self.ctrPanel.grid(row=5, column=1, columnspan=2, sticky=W + E)

        self.prevBtn=Button(self.ctrPanel, text='<< Prev', width=10, command=self.prevImage)
        self.prevBtn.pack(side=LEFT, padx=5, pady=3)

        self.nextBtn=Button(self.ctrPanel, text='Next >>', width=10, command=self.nextImage)
        self.nextBtn.pack(side=LEFT, padx=5, pady=3)

        self.progLabel=Label(self.ctrPanel, text="Progress:     /    ")
        self.progLabel.pack(side=LEFT, padx=5)

        self.tmpLabel=Label(self.ctrPanel, text="Go to Image No.")
        self.tmpLabel.pack(side=LEFT, padx=5)

        self.idxEntry=Entry(self.ctrPanel, width=5)
        self.idxEntry.pack(side=LEFT)

        self.goBtn=Button(self.ctrPanel, text='Go', command=self.gotoImage)
        self.goBtn.pack(side=LEFT)

        # display mouse position
        self.disp=Label(self.ctrPanel, text='')
        self.disp.pack(side=RIGHT)

        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(4, weight=1)

        # for debugging

    ##        self.setImage()
    ##        self.loadDir()

    def loadDir(self, dbg=False):
        if not dbg:
            s=self.entry.get()
            self.parent.focus()
            self.category=int(s)
        else:
            s=r'D:\workspace\python\labelGUI'
        ##        if not os.path.isdir(s):
        ##            tkMessageBox.showerror("Error!", message = "The specified dir doesn't exist!")
        ##            return
        # get image list
        self.imageDir=os.path.join(r'./Images', '%03d' % (self.category))
        self.imageList=glob.glob(os.path.join(self.imageDir, '*.jpeg'))
        if len(self.imageList) == 0:
            print 'No .jpeg images found in the specified dir!'
            return

        # default to the 1st image in the collection
        self.cur=1
        self.total=len(self.imageList)

        # set up output dir
        self.outDir=os.path.join(r'./Labels', '%03d' % (self.category))
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)


        self.loadImage()
        print '%d images loaded from %s' % (self.total, s)

    def loadImage(self):
        # load image
        imagepath=self.imageList[self.cur - 1]
        self.img=Image.open(imagepath)
        r=min(SIZE[0] / float(self.img.size[0]), SIZE[1] / float(self.img.size[1]))
        r=0.5
        new_size_target=int(r * self.img.size[0]), int(r * self.img.size[1])
        self.img=self.img.resize(new_size_target, Image.ANTIALIAS)
        self.curr_img=self.img
        self.tkimg=ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.progLabel.config(text="%04d/%04d" % (self.cur, self.total))

        # load labels
        self.clearBBox()
        self.imagename=os.path.split(imagepath)[-1].split('.')[0]
        labelname=self.imagename + '.txt'
        self.labelfilename=os.path.join(self.outDir, labelname)
        bbox_cnt=0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    if i == 0:
                        bbox_cnt=int(line.strip())
                        continue
                    tmp=[int(t.strip()) for t in line.split()]
                    ##                    print tmp
                    self.bboxList.append(tuple(tmp))
                    tmpId=self.mainPanel.create_rectangle(tmp[0], tmp[1], \
                                                          tmp[2], tmp[3], \
                                                          width=2, \
                                                          outline=COLORS[(len(self.bboxList) - 1) % len(COLORS)])
                    self.bboxIdList.append(tmpId)
                    self.listbox.insert(END, '(%d, %d) -> (%d, %d)' % (tmp[0], tmp[1], tmp[2], tmp[3]))
                    self.listbox.itemconfig(len(self.bboxIdList) - 1,
                                            fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])

    def saveImage(self):
        with open(self.labeflfilename, 'w') as f:
            f.write('%d\n' % len(self.bboxList))
            for bbox in self.bboxList:
                f.write(' '.join(map(str, bbox)) + '\n')
        print 'Image No. %d saved' % (self.cur)

    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y']=event.x, event.y
        else:
            x1, x2=min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2=min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2))
            self.bboxIdList.append(self.bboxId)
            self.bboxId=None
            self.listbox.insert(END, '(%d, %d) -> (%d, %d)' % (x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg=COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click']=1 - self.STATE['click']

    def mouseMove(self, event):
        self.disp.config(text='x: %d, y: %d' % (event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl=self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width=2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl=self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width=2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId=self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], event.x, event.y, width=2,
                                                        outline=COLORS[len(self.bboxList) % len(COLORS)])

    def mouseWheel(self, event):
        print(event)
        if event.num == 4:
            self.scale*=1.1
        elif event.num == 5:
            self.scale*=0.9
        self.redraw(event.x, event.y)

    def move_from(self, event):
        print(event)
        self.mainPanel.scan_mark(event.x, event.y)
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def move_to(self, event):
        print(event)
        self.mainPanel.scan_dragto(event.x, event.y, gain=1)
        # self.x = self.mainPanel.canvasx(event.x)
        # self.y = self.mainPanel.canvasy(event.y)
        #
        # x_diff = self.pan_start_x - event.x
        # y_diff = self.pan_start_y - event.y
        #
        # self.mainPanel.move(x_diff,y_diff)
        # self.mainPanel.scale(xscale = 1.0, yscale = 1.0, xoffset = x_diff, yoffset = y_diff)



        #self.mainPanel.scale(ALL, x, y, self.scale, self.scale)
    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId=None
                self.STATE['click']=0

    def delBBox(self):
        sel=self.listbox.curselection()
        if len(sel) != 1:
            return
        idx=int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList=[]
        self.bboxList=[]

    def resetImage(self):
        self.tkimg=ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
        self.curr_img=self.img
        self.bright_img=None
        self.contrast_img=None
        self.sharp_img=None

    def prevImage(self, event=None):
        self.saveImage()
        if self.cur > 1:
            self.cur-=1
            self.loadImage()

    def nextImage(self, event=None):
        self.saveImage()
        if self.cur < self.total:
            self.cur+=1
            self.loadImage()

    def gotoImage(self):
        idx=int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur=idx
            self.loadImage()

    def adjBright(self, bright_value):
        self.bright_value=float(bright_value)

        if self.contrast_img != None:
            self.curr_img=self.contrast_img
        elif self.sharp_img != None:
            self.curr_img=self.sharp_img

        self.bright_img=ImageEnhance.Brightness(self.curr_img).enhance(bright_value)
        self.tkimg=ImageTk.PhotoImage(self.bright_img)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)
        # self.curr_img = self.bright_img

    def adjContrast(self, contrast_value):
        contrast_value=float(contrast_value)

        if self.bright_img != None:
            self.curr_img=self.bright_img
        elif self.sharp_img != None:
            self.curr_img=self.sharp_img

        self.contrast_img=ImageEnhance.Contrast(self.curr_img).enhance(contrast_value)
        self.tkimg=ImageTk.PhotoImage(self.contrast_img)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)

    def adjSharp(self, sharp_value):
        sharp_value=float(sharp_value)

        if self.contrast_img != None:
            self.curr_img=self.contrast_img
        elif self.bright_img != None:
            self.curr_img=self.bright_img

        self.sharp_img=ImageEnhance.Sharpness(self.curr_img).enhance(sharp_value)
        self.tkimg=ImageTk.PhotoImage(self.sharp_img)
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)

    def redraw(self, x=0, y=0):

        iw, ih= self.curr_img.size
        size=int(iw * self.scale), int(ih * self.scale)
        self.tkimg=ImageTk.PhotoImage(self.curr_img.resize(size))
        self.mainPanel.config(width=max(self.tkimg.width(), 400), height=max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image=self.tkimg, anchor=NW)

        #self.img_id=self.canvas.create_image(x, y, image=self.img)

        # tell the canvas to scale up/down the vector objects as well
        self.mainPanel.scale(ALL, x, y, self.scale, self.scale)


##    def setImage(self, imagepath = r'test2.png'):
##        self.img = Image.open(imagepath)
##        self.tkimg = ImageTk.PhotoImage(self.img)
##        self.mainPanel.config(width = self.tkimg.width())
##        self.mainPanel.config(height = self.tkimg.height())
##        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)

if __name__ == '__main__':
    root=Tk()
    tool=LabelTool(root)
    root.resizable(width=True, height=True)
root.mainloop()
