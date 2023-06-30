# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:11:55 2021

@author: Michael H. Udin
"""

#%% Imports
import glob
import os
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor

#%% Variables
setname1 = 'InputSetName' #input set name
setname2 = 'OutputSetName' #output set name
qq=glob.glob('ThePathToSet/' + setname1 + '/*.TheImageExtension') #the folder you need to
opath = 'E:/Sets/' + setname2 + '/'
os.makedirs(opath, exist_ok=True) #creates spath directory if doesn't exist

#%% Some math
oldsize = 256 #dimensions of input images e.g. 256 for 256x256
newsize = 150 #dimensions of input images e.g. 150 for 150x150
adj1=int((oldsize-newsize)/2) #calculates crop amount
adj2=int(newsize+adj1) #secondary crop variable

#%% Fast cropping with threading
def crop_it(image):
    Image.open(image).crop((adj1, adj1, adj2, adj2)).save(opath+os.path.basename(image))

start_time = time.time()
with ThreadPoolExecutor(20) as executor:
    {executor.submit(crop_it, image) for image in qq}
print(f"--- %.4f seconds ---" % (time.time() - start_time))
