# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:15:56 2021

@author: Michael H. Udin
"""
#%% Timer Tic
import time
start_time = time.time()

#%% Main imports
import pydicom as pd
import glob
from pydicom import dcmread
from PIL import Image, ImageOps
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
import numpy as np
import cv2

#%% Major variables
setname = 'xmde1' #modify output set name here
spath = 'E:/Sets/' + setname + '/'
makedirs(spath, exist_ok=True) #creates spath directory if doesn't exist
flist = [r"E:\DICOM Main\Redata\Normal\*\DICOMOBJ\*", #noscar
          r"E:\DICOM Main\Redata\Control\*\DICOMOBJ\*", #noscar
          r"E:\DICOM Main\Redata\IMS\*\DICOMOBJ\*", #scar
          r"E:\DICOM Main\Redata\Acute*\*\DICOMOBJ\*"] #scar
# flist = [r"C:\Users\Pesto\Desktop\Redata\Control\*\DICOMOBJ\*"]
f='.png' #output format
scar = 'blank'
namelist = [ 'SAX MDE',
 'Sax 2D MDE',
 'Sax 2D MDE#2',
 'Sax 2D MDE  TI 600',
 'SAX MDE +gad',
 'SAX MDE BH',
 'SAX 2D MDE']

#%% Retrieve specified image data, crop, and save as png in designated location
zero2= ['00'] * 10
zero1 = ['0'] * 90
zero0 = [''] * 900
zero = zero2 + zero1 + zero0
st1 = [] #start for noscar
en1 = [] #end for noscar
st2 = [] #start for scar
en2 = [] #end for scar

for i in flist:
    print(i)
    scarchk = scar
    # scar='noscar'
    if i == flist[0]:
        scar = 'noscar'
    if i == flist[1]:
        scar = 'noscar'
    if i == flist[2]:
        scar = 'scar'
    if i == flist[3]:
        scar = 'scar'
    ns=0 #number of image in subset tracker
    ts=0 #time of scan tracker
    pid=0 #patient id
    sub=0 #subset tracker    
    if scar != scarchk:
        n=0 #total item count
        m=0 #subject number
    for file in glob.glob(i):
        apple = pd.dcmread(file)
        t = [0x8, 0x103e] in apple # type of scan check
        a = [0x20, 0x1002] in apple # number of images in acquisition check
        s = [0x20, 0x13] in apple # number of image in subset check
        if t == True and a == True and s == True:
            tt = apple[0x8, 0x103e].value # type of scan value        
            if tt in namelist:
                print(tt)
                aa = apple[0x20, 0x1002].value # number of images in acquisition value                
                ss = apple[0x20, 0x13].value # number of image in subset value 
                ti = apple[0x9, 0x10e9].value # time of scan
                if ti != ts:
                    if ts == 0:
                        pid = apple[0x10, 0x20].value # patient id
                        m+=1
                    pod = apple[0x10, 0x20].value # patient id
                    if ns != 0:
                        sb = zero[m] + str(m)
                        print(f"Subset {sb} part {sub} has {ns} images.")                        
                    if pid == pod:
                        sub+=1
                    else:
                        sub=1
                        m+=1
                    ns=0
                    ts = ti
                if ss > ns:
                    ns=ss
                pid = apple[0x10, 0x20].value # patient id
                mmm = zero[m] + str(m) #subject number
                sss = zero[ss] + str(ss) #number of image in subset
                n+=1                               
                ds = dcmread(file)
                img=ds.pixel_array
                winw = apple[0x28, 0x1051].value # type of scan value
                # scaled_img = cv2.convertScaleAbs(img-np.min(img), alpha=(255.0 / min(np.max(img)-np.min(img), 10000)))
                scaled_img = cv2.convertScaleAbs(img, alpha=(255.0/winw)) #adjusts pixel intensity using window width
                dst = spath + scar + mmm + '-' + str(sub) + '-' + sss + f
                cv2.imwrite(dst, scaled_img)
    sb = zero[m] + str(m)
    print(f"Subset {sb} part {sub} has {ns} images.")                    
    print(f"Total number of images in set: {n}")
        
#%% Timer Toc                        
dur=time.time() - start_time                     
print(f"*** Finished in {dur:.4f} seconds ***")