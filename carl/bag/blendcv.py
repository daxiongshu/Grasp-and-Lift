import numpy as np
import pandas as pd
from sklearn import metrics
from glob import glob
import sys

def myauc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)

names=['HandStart','FirstDigitTouch','BothStartLoadPhase','LiftOff','Replace','BothReleased']

subjects = range(1,13)
real=[]
for subject in subjects:
    fnames =  ['../../data/train/subj%d_series%d_events.csv' % (subject,i) for i in range(7,9)]
    for fname in fnames:
        labels= pd.read_csv(fname)
        real.append(labels)
        print fname,labels.shape
real=pd.concat(real)
print 'combined', real.shape
real['subject']=real['id'].apply(lambda x:x.split('_')[0][4:])
real.drop('id',inplace=True,axis=1)
xx=[]
bestweight=[]
bestscore=[]
start={}
for i in range(1,13):
    start[i]=0

pbest=pd.read_csv('../vali-stack_srk1/vali1_3_new_cv.csv') # just a placeholder, could be any other file with the same dimension
for name in names:
    pbest[name]=0.0
import os
for fname in os.listdir('cv'):
    if '.csv' not in fname:
        continue
    
    sub2=pd.read_csv('cv/'+fname)
    
    subject=str(fname.split('_')[1][3:])
    mask=np.array(real['subject']==subject)
    if start[int(subject)]==0:
        start[int(subject)]=1        
        for name in names:
            tmp=np.array(pbest[name])
            tmp[mask]=np.array(sub2[name])
            pbest[name]=tmp
        continue
    for name in names:
    
        y=np.array(real[name])
        y=y[mask]
        yr1=np.array(pbest[name])[mask]
        yr2=np.array(sub2[name])
        tmp=np.array(pbest[name])
        bests=-1
        bestf=-1
        besty=None
        for j in range(11):
            yr=yr1*j+yr2*(10-j)
            yr/=10
            m=myauc(y,yr)
            print name,j,m
            if bests<m:
                bests=m
                bestf=j
                besty=yr
        tmp[mask]=besty
        pbest[name]=tmp
        print subject,name,bests
        
    


pbest.to_csv('allcv.csv',index=False)
