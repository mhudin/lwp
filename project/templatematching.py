# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:27:24 2023

@author: Michael H. Udin
"""

#%% Timer
import time
start_time = time.time()

#%% Imports
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

#%% Path Info
qq = glob.glob('C:/NegativeTemplateSet/*.png') #source folder for negative cases
rr = glob.glob('C:/PositiveTemplateSet/*.png') #source folder for positive cases
bb = glob.glob('C:/ExternalDataset/*.png') #source folder for external testing dataset

#%% Setup Variables
tset = 0.73 #set threshold
negative = 'noscar' #replace 'noscar' with name convention for negative images

#%% Create patient-level labels and counting number of files per patient
Z=0 #patient number counter
N=[] #list of number of images per patient
S=[] #patient-level labels

mismatch='x'
num=0

for file in bb:    
    match=os.path.basename(file)[:-10] #adjust [:-10] based on file names used
    if file == bb[0]:
        mismatch=match
    if match != mismatch:
        Z+=1
        S.append(0) if negative in mismatch else S.append(1)
        N+=[num]
        num=0
    mismatch=match
    num+=1
    if file == bb[-1]:
        Z+=1
        S.append(0) if negative in mismatch else S.append(1)
        N+=[num]

print('Number of patients:', Z)
print('Number of files:', len(bb))

#%% Create single-image labels
SS = [0 if negative in os.path.basename(file)[:-10] else 1 for file in bb] # forifelse in one line

#%% Define template-matching function
def match_it(gray, image):
    return(cv2.matchTemplate(cv2.imread(gray,0), cv2.imread(image,0), cv2.TM_CCOEFF_NORMED))

#%% Loop Variables
tpc=0
fnc=0
tnc=0
fpc=0
undin=0
undip=0
undpn=0
undpp=0

goodn = 0
goodp = 0
badn = 0
badp = 0

count=0
patnum=0

for num in N:
    tp=0
    fn=0
    tn=0
    fp=0
    for patim in range(num):
        gray = bb[count] #image for matching
        
        # Vote counters
        countyes=0
        countno=0
        
        # Threshold
        trsh=[tset]
        
        for thrsh in trsh:  
            # Negative case counter
            val = []
            with ThreadPoolExecutor(20) as executor:
                results = {executor.submit(match_it, gray, image) for image in qq}
                for result in as_completed(results):
                    val.append(result.result())
            countno = len([i for i in val if i >= thrsh])
    
            # Positive case counter
            val2 = []
            with ThreadPoolExecutor(20) as executor:
                results2 = {executor.submit(match_it, gray, image) for image in rr}
                for result in as_completed(results2):
                    val2.append(result.result())
            countyes = len([i for i in val2 if i >= thrsh])

        countno=countno/len(qq)
        countyes=countyes/len(rr)

        chrk=negative in gray
        if countyes < countno and chrk == True:
            tn+=1
            print(f'true negative: {gray}')
            goodn+=1
        elif countyes < countno and chrk == False:
            fn+=1
            print(f'false negative: {gray}')
            badn+=1
        if countyes > countno and chrk == True:
            fp+=1
            print(f'false positive: {gray}')
            badp+=1
        elif countyes > countno and chrk == False:
            tp+=1
            print(f'true positive: {gray}')
            goodp+=1
        if countyes == countno:
            if chrk == True:
                undin+=1
                print(f'undecided negative: {gray}')
            else:
                undip+=1
                print(f'undecided positive: {gray}')

        if patim == num - 1 and S[patnum] == 0:
            if tn > fp:
                tnc+=1
            if fp > tn:
                fpc+=1
            if fp == tn:
                undpn+=1
        if patim == num - 1 and S[patnum] == 1:
            if tp > fn:
                tpc+=1
            if fn > tp:
                fnc+=1
            if fn == tp:
                undpp+=1
    
        count+=1
        
    patnum+=1

#%% Calculate and report metrics
iacc = (goodn+goodp)/(goodn+goodp+badn+badp+undin+undip)
pacc = (tpc+tnc)/(tpc+tnc+fpc+fnc+undpn+undpp)
print('')
print(f'Threshold: {thrsh}')
print('')
print(f'True positives: {tpc}')
print(f'False negatives: {fnc}')
print(f'True negatives: {tnc}')
print(f'False positives: {fpc}')
print('')
print(f'patient acc: {pacc}')
print('')
print(f'individual acc: {iacc}')
print('')
tp=goodp
fn=badn
tn=goodn
fp=badp
print(f'True positives: {tp}')
print(f'False negatives: {fn}')
print(f'True negatives: {tn}')
print(f'False positives: {fp}')
sens = tp/(tp+fn)
spec = tn/(tn+fp)
print('Sensitivity/Recall: %.2f%%' % (sens * 100.0))
print('Specificity: %.2f%%' % (spec * 100.0))
prec = tp/(tp+fp)
print('Precision: %.2f%%' % (prec * 100.0))
f1 = (2*prec*sens)/(prec+sens)
print('F1-score: %.2f%%' % (f1 * 100.0))
print('')
print('')
print(f'Undecided negative images: {undin}')
print(f'Undecided positive images: {undip}')
print(f'Undecided negative patients: {undpn}')
print(f'Undecided positive patients: {undpp}')
print('')
    
#%% Timer                                
print(f'--- {time.time() - start_time:.4f} seconds ---')
