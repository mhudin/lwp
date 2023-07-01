# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:05:51 2021

@author: Michael H. Udin
"""
#%% Timer
import time
start_time = time.time()

#%% Imports
from tensorflow.keras.applications.resnet50 import ResNet50 as RN
from tensorflow.keras.applications.resnet_v2 import ResNet152V2 as RN2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
import tensorflow as tf
import cv2
import glob
import numpy as np    
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix
import psutil
import os
import multiprocessing
import pandas as pd
import logging
import scipy.stats as st
import copy

#%% Memory control
apple = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(apple[0], True)

#%% TF output control (hides non-critical output)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
logging.getLogger('tensorflow').setLevel(logging.FATAL)

#%% Input parameters – Less than 30 minutes with ES (YTMV)
# Path information
setname = 'LWPtest2' #which set to draw images from
files = glob.glob('E:/Sets/' + setname + '/*.png')

# Machine learning variables
size, _ = cv2.imread(files[0], 0).shape # extract image size (size of one dimension, e.g. 512 for a 512x512)
ep1 = 500 #number of epochs
split = 20 #number of folds
rss = 24 #set number for random number generator
bat = 64 #batch size – higher numbers should be paired with higher learning rates
insh = (size, size, 1) #input shape (size, size, 3) or (size, size, 1) for greyscale
opt = RMSprop(learning_rate=0.0005) #set learning rate, RMSprop default = 0.001
useModel = RN #pick which CNN to use (RN=ResNet50, RN2=Resnet152V2)

#%% Callback settings
# MC is below because of required dynamic savespot

RL = ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.95,
    patience=20, min_lr=0.00005) #30

ES = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0,
    patience=100, #80
    verbose=1,
    mode='auto',
    restore_best_weights=True)

choice = [1,1,0] #[MC,RL,ES] 0=off, 1=on

#%% Allocations & Call Groups
mm=0
mn1=[]
mn2=[]
mn3=[]
sen=[]
spe=[]
pre=[]
f1s=[]
ac=[]
roc=[]
xt=[]
yt=[]
tprs=[]
fprs=[]
gain=0
memtr=[]
zero = ['0'] * 10 + [''] * 90
model=[]

#%% Subprocess List
def train1(X_test, Y_test, X_train, Y_train, q, count, class_weights, savespot, choice, MC):
    
    cblist = []
    if choice[0] == 1: #ModelCheckpoint
        cblist.append(MC)
    if choice[1] == 1: #ReduceLROnPlateau
        cblist.append(RL)
    if choice[2] == 1: #EarlyStopping
        cblist.append(ES)
    
    model = useModel(weights=None, include_top=False, input_shape=insh)
    for layer in model.layers:
     	layer.trainable = False #output as (model.layers[-1].output)
    qq = Flatten()(model.layers[-1].output)
    qq = Dense(2000, activation='relu', kernel_initializer='he_uniform')(qq)
    qq = Dropout(0.1)(qq)
    qq = Dense(1, activation='sigmoid')(qq)
    model = Model(inputs=model.inputs, outputs=qq)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, class_weight=class_weights, epochs=ep1, validation_data=(X_test,Y_test),
                        batch_size=bat, verbose=1, callbacks = cblist)
    
    # Choose action based on whether or not MC is on
    if choice[0] == 1:
        model.load_weights(savespot) # load the best weight
    else:
        model.save(savespot) # save the whole model
        
    pred=model.predict(X_test)
    
    #Output variables
    q.put(pred)

#%% ROC Creator
def rocCreator(fprs,tprs): 
                   
    idx_max = fprs.index((max(fprs, key=len)))
    fpr_max = fprs[idx_max]
        
    fpr_interp = list()
    tprs_interp = list()
    
    for b in range(len(fprs)):
        fpr_temp =  interp1d(fprs[b],tprs[b])
        fpr_interp.append(fpr_temp) 
        
        tpr_temp = fpr_interp[b](fpr_max)
        tprs_interp.append(tpr_temp)
        
    tpr_stack =  np.stack(tprs_interp,axis=0)
    tpr_std = np.std(tpr_stack, axis=0)
    #/np.sqrt(np.size(tpr_stack,0))
    tpr_avg = np.mean(tpr_stack ,axis=0)
        
    avg_roc_auc = auc(fpr_max, tpr_avg)
    
    plt.figure(figsize=(8, 8), dpi= 160, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 22})
    plt.gca().set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel='False Positive Rate', ylabel='True Positive Rate')
    
    tpr_avg = np.append([0],tpr_avg)
    fpr_max = np.append([0],fpr_max)
    tpr_std = np.append([0],tpr_std)
    
    plt.plot(fpr_max, tpr_avg, label='Mean curve, AUC: {:.4f}'.format(avg_roc_auc))
    plt.fill_between(fpr_max, tpr_avg+tpr_std, tpr_avg-tpr_std, alpha=0.5)

    plt.plot([0, 1], ls="--")

    plt.legend(loc='best')
    plt.show()

#%% Find unique patients and create labels patient labels
# Create list of base filenames
S = [os.path.basename(file)[:-10] for file in files] #create list of base filenames

# Keep the unique patients only
V=[]
for x in S:
    if x not in V:
        V.append(x)

# Create labels for unique patients
W = [0 if 'noscar' in val else 1 for val in V]

# Display as a check
print(V)
print(W)

# Prepare for use in stratified shuffle split
V = np.array(V)
W = np.array(W)

#%% Main loop
# Create stratified shuffle split for x splits(folds)
from sklearn.model_selection import StratifiedShuffleSplit    
sss = StratifiedShuffleSplit(n_splits=split, test_size=0.20, random_state=rss)
sss.get_n_splits(V, W)

count=0

# The main loop
for train_index, test_index in sss.split(V, W):
    
    fold_time = time.time()

    if __name__ == '__main__':
        count+=1 #counter here for naming purposes
        
        if count == 1: print('')
        print(f"<---- Processing split {count} ---->")
        print('')

        print('ResNet CNN is now:', __name__)
        print('')
        
        V_train, V_test = V[train_index], V[test_index]
        W_train, W_test = W[train_index], W[test_index]
    
        X_train = []
        X_test = []
        Y_train = []
        Y_test = []
        
        # Create X_train, Y_train, X_test, Y_test
        for file in files:
            image = cv2.imread(file, 0) #in (file, 0) the , 0  reads as grayscale!
            nm = os.path.basename(file)[:-10]
            QQ = 0 if 'noscar' in os.path.basename(file) else 1
            
            if nm in V_train:
                X_train.append(image)
                Y_train.append(QQ)
            else:
                X_test.append(image)
                Y_test.append(QQ)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        Y_train = np.array(Y_train)
        Y_test = np.array(Y_test)
        
        # Calculate class balance and create class weights    
        Q=np.sum(Y_train == 0)/len(Y_train)
        P=1-Q
        Q=round(Q*100, 3)
        P=round(P*100, 3)
        print(f"Train ratio: {Q} / {P}")
        Q2=np.sum(Y_test == 0)/len(Y_test)
        P2=1-Q2
        Q2=round(Q2*100, 3)
        P2=round(P2*100, 3)
        print(f"Test ratio: {Q2} / {P2}")
        print('')
        
        cw1=round(((50-Q)/50)+1,5)
        cw2=round(((50-P)/50)+1,5)
         
        print(f"Class 0 Weight: {cw1}")
        print(f"Class 1 Weight: {cw2}")
        print('')
        
        class_weights = {0: cw1, 1: cw2}  
        
        if choice[0] == 1: #ModelCheckpoint
            savespot=setname+zero[count]+str(count)+'weight.h5' #update savespot
        else:
            savespot=setname+zero[count]+str(count)+'model.h5' #update savespot
        
        MC = ModelCheckpoint(
            filepath=savespot,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
    
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=train1, args=(X_test, Y_test, X_train, Y_train, q,
                                                          count, class_weights, savespot, choice, MC))
        p.start()
        
        # Retrieve variables from inside multiprocessing process  
        pred = q.get()
    
        print('')       
        p.join()
        p.close()

    if __name__ == '__main__':
        # Performance metric calculation
        predZ=np.array(pred)  
        predZ = [0 if nm < 0.5 else 1 for nm in predZ] # convert float preds to binary
        tn, fp, fn, tp = confusion_matrix(Y_test, predZ).ravel() #confusion matrix to get tn, fp, fn, tp
        
        print(f"True negatives: {tn}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"True positives: {tp}")
        print('')
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        print(f"Sensitivity/Recall: {sens}")
        print(f"Specificity: {spec}")
        print('')
        prec = tp/(tp+fp)
        print(f"Precision: {prec}")
        f1 = (2*prec*sens)/(prec+sens)
        print(f"F1 score: {f1}")
        print('')
        macc = (tp+tn)/(tp+tn+fp+fn)
        print('Accuracy: %.3f%%' % (macc * 100.0))
        print('')
        sen.append(sens)
        spe.append(spec)
        pre.append(prec)
        f1s.append(f1)
        ac.append(macc)
        if macc > mm:
            mm = macc

    if __name__ == '__main__':
        # Get and append false positive and true positive rates
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Y_test, pred)
        rocs=roc_auc_score(Y_test, pred)
        print(f"ROC AUC: {rocs}")
        roc.append(rocs)
        fprs.append(list(false_positive_rate1))
        tprs.append(list(true_positive_rate1))
        print('')
        print('')
        print(f'Fold accuracy: {ac}')
        print('')
        print('')
        
        # Monitor video memory usage
        py = psutil.Process(os.getpid())
        print('VRAM usage: {} GB'.format(round(py.memory_info()[0]/2. ** 30,4)))
        memtr.append(round(py.memory_info()[0]/2. ** 30,4))
        print('')

        # (table) Output tables periodically
        tac = [round(elem, 4) for elem in ac]
        bc = list(range(1,len(tac)+1))
        bc.append('AVG')
        stmac = round(np.mean(tac), 4)
        tac.append(stmac)
        
        # (table) Convert to arrays and make vertical
        tac = np.array(tac)
        tac = tac.reshape(len(tac), 1)
        bc = np.array(bc)
        bc = bc.reshape(len(bc), 1)
        
        # (table) Stack together vertically
        cc = np.column_stack((bc, tac))
        
        # (table) Make table informtion into Pandas dataframe
        df = pd.DataFrame(cc, columns=['Run #', 'Value'])
        
        # (table) Prepare for table display
        fig, ax = plt.subplots(figsize=(20,20))
        fig.tight_layout()
        fig.patch.set_visible(False)
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        table.scale(1,10)
        table.auto_set_font_size(False)
        table.set_fontsize(50) #80
        ax.axis('off')
        cells = table.properties()["celld"]
        
        # (table) Build and show table
        for i in range(0, len(tac)+1):
              cells[i, 0].set_text_props(ha="center")
              cells[i, 0].get_text().set_color('white')
              cells[i, 0].set_facecolor("#000000")
              cells[i, 0].set_edgecolor('white')
              cells[i, 1].set_text_props(ha="center")
              cells[i, 1].get_text().set_color('white')
              cells[i, 1].set_facecolor("#000000")
              cells[i, 1].set_edgecolor('white')
        plt.show()
        
        # Display time taken to complate the fold
        print(f"Fold completed in %.4f seconds" % (time.time() - fold_time))
        print('')
        print('')

if __name__ == '__main__':
    # Final performance metric calculations
    rocCreator(fprs,tprs)
    senn=np.mean(sen)
    spen=np.mean(spe)
    pren=np.mean(pre)
    f1sn=np.mean(f1s)
    rocn=np.mean(roc)
    mn=np.mean(ac)
    dsen=np.std(sen)
    dspe=np.std(spe)
    dpre=np.std(pre)
    df1s=np.std(f1s)
    droc=np.std(roc)
    dac=np.std(ac)
    ci = st.t.interval(confidence=0.95, df=len(roc)-1, loc=rocn, scale=st.sem(roc)) #confidence interval 95% for auroc
    ci0=ci[0]
    ci1=ci[1]    
    
    # Output final performance metrics
    print('Mean ROC AUC: %.4f' % (rocn))
    print('Mean Sensitivity: %.3f%%' % (senn * 100.0))
    print('Mean Specificity: %.3f%%' % (spen * 100.0))
    print('Mean Precision: %.3f%%' % (pren * 100.0))
    print('Mean F1 Score: %.3f%%' % (f1sn * 100.0))
    print('Mean Accuracy: %.3f%%' % (mn * 100.0))
    print('Max Accuracy: %.3f%%' % (mm * 100.0))
    print('')
    print('RECORD VALUES')
    print('Mean Accuracy ± stdev: %.2f%% ± %.2f%%' % (mn * 100.0, dac * 100.00))
    print('Mean Sensitivity ± stdev: %.2f%% ± %.2f%%' % (senn * 100.0, dsen * 100.00))
    print('Mean Specificity ± stdev: %.2f%% ± %.2f%%' % (spen * 100.0, dspe * 100.00))
    print('Mean F1-score ± stdev: %.2f%% ± %.2f%%' % (f1sn * 100.0, df1s * 100.00))
    print('Mean AUROC ± stdev: %.4f ± %.4f' % (rocn, droc))
    print('AUROC 95%% confidence interval: %.4f – %.4f' % (ci0, ci1))
    print('')
    print('')
    
    #%% Timer                                
    print(f"Total time: %.4f seconds" % (time.time() - start_time))
