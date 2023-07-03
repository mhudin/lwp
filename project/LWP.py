# -*- coding: utf-8 -*-
"""
Created on Tue May 23 18:06:35 2023

@author: Michael H. Udin
"""

#%% Timer
import time
full_time = time.time()

#%% Imports
import glob
import os
import numpy as np
from shutil import copy2 as cf
from concurrent.futures import ThreadPoolExecutor
import cv2
from PIL import Image

#%% Variables
setname1 = 'xmde1' #input set name
setname2 = 'LWPout1' #output set name
setname3 = 'LWPout2' #output set name
setname4 = 'LWPout3' #output set name
setname5 = 'LWPout4' #output set name
setname6 = 'LWPout5' #output set name
qq=glob.glob('E:/Sets/' + setname1 + '/*.png')
opath1 = 'E:/Sets/' + setname2 + '/'
opath2 = 'E:/Sets/' + setname3 + '/'
opath3 = 'E:/Sets/' + setname4 + '/'
opath4 = 'E:/Sets/' + setname5 + '/'
opath5 = 'E:/Sets/' + setname6 + '/'
os.makedirs(opath1, exist_ok=True) #creates spath directory if doesn't exist
os.makedirs(opath2, exist_ok=True) #creates spath directory if doesn't exist
os.makedirs(opath3, exist_ok=True) #creates spath directory if doesn't exist
os.makedirs(opath4, exist_ok=True) #creates spath directory if doesn't exist
os.makedirs(opath5, exist_ok=True) #creates spath directory if doesn't exist

#%% Some math
oldsize = 256
newsize = 150
adj1=int((oldsize-newsize)/2) #this is calculated by (150-80)/2
adj2=int(newsize+adj1) #w+x or 150 + 53

#%% V3
def crop_it(image):
    Image.open(image).crop((adj1, adj1, adj2, adj2)).save(opath1+os.path.basename(image))
    
def bright_it1(image):
    cv2.imwrite(opath2+os.path.basename(image),(cv2.multiply(cv2.imread(image,0),255/(cv2.imread(image,0).max()))))
    
def bright_it2(image):
    cv2.imwrite(opath5+os.path.basename(image),(cv2.multiply(cv2.imread(image,0),255/(cv2.imread(image,0).max()))))

crop150_time = time.time()
with ThreadPoolExecutor(20) as executor:
    {executor.submit(crop_it, image) for image in qq}
print(f"--- %.4f seconds ---" % (time.time() - crop150_time))


qq = glob.glob(opath1+'*.png')


bright1_time = time.time()
with ThreadPoolExecutor(20) as executor:
    {executor.submit(bright_it1, image) for image in qq}
print(f"--- %.4f seconds ---" % (time.time() - bright1_time))


qq = glob.glob(opath2+'*.png')


crop80_time = time.time()
#%% Variables
fdt=0 #total number of failed circle detections
fc=0 #tracks number of files per set
thr=500 #threshold for failed circle detections
mo=40 #half size of desired image size square dimension
imsz=150 #input image size

#%% Variable holders
stnm='' #holder
a=[] #holder
b=[] #holder
bnh='blank' #holder
fd=0 #holder
mma=[] #holder
mmb=[] #holder
mmma=[] #holder
mmmb=[] #holder
meda=[] #holder
medb=[] #holder
fl=[] #holder for file list in set
tfc=0 #holder

#%% New variables 05232023
rotlist=[0,90,180,270] #list of rotations

#%% Define circle detector
def detect_it(iset, varset, a, b, fd, fdt):
    count=0
    for img in iset: #parameter set 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayscale
        gray_blurred = cv2.blur(gray, (3, 3)) #blurs
        blur = cv2.blur(img, (3, 3)) #blurs
        detected_circles = cv2.HoughCircles(gray_blurred,
                            cv2.HOUGH_GRADIENT, 0.9, 150, param1 = 50,
                        param2 = varset[0], minRadius = varset[1], maxRadius = 30) #26 12 26
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles)) #get circle params a, b, r as ints
            for pt in detected_circles[0, :]:
                va, vb, vr = pt[0], pt[1], pt[2]  
                if count == 0:
                    a.append(va)
                    b.append(vb)
                elif count == 1:
                    a.append(vb)
                    b.append(imsz-va)
                elif count == 2:
                    a.append(imsz-va)
                    b.append(imsz-vb)
                elif count == 3:
                    a.append(imsz-vb)
                    b.append(va)
                count+=1
        else:
            fd+=1
            fdt+=1
        
    return a, b, fd, fdt

def detect_it2(iset, varset, a, b, fd, fdt):
    count=0
    for img in iset: #parameter set 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayscale
        gray_blurred = cv2.blur(gray, (3, 3)) #blurs
        detected_circles = cv2.HoughCircles(gray_blurred,
                            cv2.HOUGH_GRADIENT, 0.9, 150, param1 = 50,
                        param2 = varset[0], minRadius = varset[1], maxRadius = 30) #26 12 26
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles)) #get circle params a, b, r as ints
            for pt in detected_circles[0, :]:
                va, vb, vr = pt[0], pt[1], pt[2]  
                if count == 0:
                    a.append(imsz-va)
                    b.append(vb)
                elif count == 1:
                    a.append(vb)
                    b.append(va)
                elif count == 2:
                    a.append(va)
                    b.append(imsz-vb)
                elif count == 3:
                    a.append(imsz-vb)
                    b.append(imsz-va)
                count+=1
        else:
            fd+=1
            fdt+=1
        
    return a, b, fd, fdt

#%% Multiple-form Circle Detection
for file in qq:
    base = os.path.basename(file)
    bn=base[:-8] #reduce file name to base file name    
    stnm=bn #storing current file as stored file

    #file number and name tracking
    if bn != bnh:
        if file != qq[0]:
            if meda != [] and medb != []:
                mmma = mmma+[int(round(np.median(meda),0))]*fc
                mmmb = mmmb+[int(round(np.median(medb),0))]*fc
                mma = []
                mmb = []
            else:
                mmma = mmma+[75]*fc
                mmmb = mmmb+[75]*fc
            fl=[]
            fc=0
            meda=[]
            medb=[]
        bnh=bn
    
    X = []
    image = cv2.imread(file) #original file
    for i in rotlist:
        X.append(image)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    Y = []
    image2 = cv2.flip(image, 0) #flip and rotate
    for i in rotlist:
        Y.append(image2)
        image2 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)

    a = [] #x coordinate holder
    b = [] #y coordinate holder
    
    # Circle detections on original set X
    a, b, fd, fdt = detect_it(X, [24,12], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [26,12], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [28,12], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [24,14], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [26,14], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [28,14], a, b, fd, fdt)

    # Circle detections on flip set Y
    a, b, fd, fdt = detect_it2(Y, [24,12], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [26,12], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [28,12], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [24,14], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [26,14], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [28,14], a, b, fd, fdt)
   
    # Outlier fixing
    aa=[]
    bb=[]
    
    for chka in a:
        if chka > (imsz*0.625) or chka < (imsz*0.375):
            aa.append(chka)
            fd+=0.5
    for chkb in b:
        if chkb > (imsz*0.625) or chkb < (imsz*0.375):
            bb.append(chkb)
            fd+=0.5
    for redux in aa:
        a.remove(redux)
    for redux in bb:
        b.remove(redux)
    if a != []:
        for adds in a:
            meda.append(adds)
    if b != []:
        for adds in b:
            medb.append(adds)

    fd=0
    fc+=1
    fl.append(file)
    tfc+=1

    if file == qq[-1]: #last file action?
        print('last file')
        if meda != [] and medb != []:                      
            mmma = mmma + [int(round(np.median(meda),0))]*fc
            mmmb = mmmb + [int(round(np.median(medb),0))]*fc
        else:
            mmma = mmma+[75]*fc
            mmmb = mmmb+[75]*fc

#%% Cropping center
mo2=mo*2 #size of desired image
mr=mo-3 #radius of mask
rec=0 #counter
for file in qq:      
    image = cv2.imread(file,0)         
    mask = np.zeros_like(image)
    ga = int(mmma[rec])
    gb = int(mmmb[rec])
    mask = cv2.circle(mask, (ga, gb), mr, (255,255,255), -1)
    result = cv2.bitwise_and(image, mask) #applies mask
    
    p = ga-mo
    q = ga+mo
    
    result = result[p:q,p:q] #crops arrays to specified size    
    cv2.imwrite(opath3+os.path.basename(file), result) #save image to output folder
    rec+=1
    
print(f"--- %.4f seconds ---" % (time.time() - crop80_time))


qq = glob.glob(opath3+'*.png')


redux_time = time.time()
#%% Variables
fdt=0 #total number of failed circle detections
fc=0 #tracks number of files per set
mo=40 #half size of desired image size square dimension
imsz=80 #input image size

#%% Variable holders
stnm='' #holder
a=[] #holder
b=[] #holder
bnh='blank' #holder
fd=0 #holder
mma=[] #holder
mmb=[] #holder
mmma=[] #holder
mmmb=[] #holder
meda=[] #holder
medb=[] #holder
fl=[] #holder for file list in set
tfc=0 #holder
thr=13 #actual variable
remqq=[] #holder

param11=25
param12=23
param13=24
param21=25
param22=23
param23=24
maxrad=28
minrad1=13
minrad2=14

#%% New variables 05232023
rotlist=[0,90,180,270] #list of rotations

#%% Define circle detector
def detect_it(iset, varset, a, b, fd, fdt):
    count=0
    for img in iset: #parameter set 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayscale
        gray_blurred = cv2.blur(gray, (3, 3)) #blurs
        blur = cv2.blur(img, (3, 3)) #blurs
        detected_circles = cv2.HoughCircles(gray_blurred,
                            cv2.HOUGH_GRADIENT, 0.9, 150, param1 = 50,
                        param2 = varset[0], minRadius = varset[1], maxRadius = 28) #26 12 26
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles)) #get circle params a, b, r as ints
            for pt in detected_circles[0, :]:
                va, vb, vr = pt[0], pt[1], pt[2]  
                if count == 0:
                    a.append(va)
                    b.append(vb)
                elif count == 1:
                    a.append(vb)
                    b.append(imsz-va)
                elif count == 2:
                    a.append(imsz-va)
                    b.append(imsz-vb)
                elif count == 3:
                    a.append(imsz-vb)
                    b.append(va)
                count+=1
        else:
            fd+=1
            fdt+=1
        
    return a, b, fd, fdt

def detect_it2(iset, varset, a, b, fd, fdt):
    count=0
    for img in iset: #parameter set 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts to grayscale
        gray_blurred = cv2.blur(gray, (3, 3)) #blurs
        detected_circles = cv2.HoughCircles(gray_blurred,
                            cv2.HOUGH_GRADIENT, 0.9, 150, param1 = 50,
                        param2 = varset[0], minRadius = varset[1], maxRadius = 28) #26 12 26
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles)) #get circle params a, b, r as ints
            for pt in detected_circles[0, :]:
                va, vb, vr = pt[0], pt[1], pt[2]  
                if count == 0:
                    a.append(imsz-va)
                    b.append(vb)
                elif count == 1:
                    a.append(vb)
                    b.append(va)
                elif count == 2:
                    a.append(va)
                    b.append(imsz-vb)
                elif count == 3:
                    a.append(imsz-vb)
                    b.append(imsz-va)
                count+=1
        else:
            fd+=1
            fdt+=1
        
    return a, b, fd, fdt

#%% Multiple-form Circle Detection with removal
for file in qq:
    base = os.path.basename(file)
    bn=base[:-8] #reduce file name to base file name    
    stnm=bn #storing current file as stored file

    #file number and name tracking
    if bn != bnh:
        if file != qq[0]:
            if meda != [] and medb != []:
                mmma = mmma+[int(round(np.median(meda),0))]*fc
                mmmb = mmmb+[int(round(np.median(medb),0))]*fc
                mma = []
                mmb = []
            else:
                mmma = mmma+[75]*fc
                mmmb = mmmb+[75]*fc
            if len(mmma)+len(mma) != tfc:
                print('')
            fl=[]
            fc=0
            meda=[]
            medb=[]
        bnh=bn
    
    X = []
    image = cv2.imread(file) #original file
    for i in rotlist:
        X.append(image)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    Y = []
    image2 = cv2.flip(image, 0) #flip and rotate
    for i in rotlist:
        Y.append(image2)
        image2 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)

    a = [] #x coordinate holder
    b = [] #y coordinate holder

    # Circle detections on original set X
    a, b, fd, fdt = detect_it(X, [23,13], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [24,13], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [25,13], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [23,14], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [24,14], a, b, fd, fdt)
    a, b, fd, fdt = detect_it(X, [25,14], a, b, fd, fdt)

    # Circle detections on flip set Y
    a, b, fd, fdt = detect_it2(Y, [23,13], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [24,13], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [25,13], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [23,14], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [24,14], a, b, fd, fdt)
    a, b, fd, fdt = detect_it2(Y, [25,14], a, b, fd, fdt)
    
    # Outlier fixes
    aa=[]
    bb=[]
    for chka in a:
        if chka > (imsz*0.625) or chka < (imsz*0.375):
            aa.append(chka)
            fd+=0.5
    for chkb in b:
        if chkb > (imsz*0.625) or chkb < (imsz*0.375):
            bb.append(chkb)
            fd+=0.5
    for redux in aa:
        a.remove(redux)
    for redux in bb:
        b.remove(redux)
    if a != []:
        for adds in a:
            meda.append(adds)
    if b != []:
        for adds in b:
            medb.append(adds)

    if fd >= thr:
        remqq.append(file)

    fd=0
    fc+=1
    fl.append(file)
    tfc+=1

    if file == qq[-1]: #last file action?
        print('last file')
        if meda != [] and medb != []:                      
            mmma = mmma + [int(round(np.median(meda),0))]*fc
            mmmb = mmmb + [int(round(np.median(medb),0))]*fc
        else:
            mmma = mmma+[75]*fc
            mmmb = mmmb+[75]*fc
    
for file in remqq:
    qq.remove(file)

#%% Moving center
rec=0
for file in qq:      
    cf(file, opath4+os.path.basename(file))
    rec+=1

#%% Timer                                
print(f"--- %.4f seconds ---" % (time.time() - redux_time))


qq = glob.glob(opath4+'*.png')


bright2_time = time.time()
with ThreadPoolExecutor(20) as executor:
    {executor.submit(bright_it2, image) for image in qq}
print(f"--- %.4f seconds ---" % (time.time() - bright2_time))


#%% Timer
print(f"--- Full script complete in %.4f seconds ---" % (time.time() - full_time))
