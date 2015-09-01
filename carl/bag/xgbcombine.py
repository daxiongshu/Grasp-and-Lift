import numpy as np
import pandas as pd
from sklearn import metrics
from glob import glob
import sys
from xgb_classifier import xgb_classifier
def myauc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)
def train_predict(X,y,Xt,yt=[],c=1):
    if c==1:
        #clf=xgb_classifier(num_round=45,eta=0.1,min_child_weight=5,depth=10, subsample=0.5,col=1) 
        clf=xgb_classifier(num_round=45,eta=0.1,min_child_weight=20,depth=20, subsample=0.1,col=0.7)
        return clf.train_predict(X,y,Xt,yt)
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

pbest=pd.read_csv('../persub/sofar.csv')
pbestsub=pd.read_csv('../vali-stack_srk1/vali1_3_new_sub.csv')
pbestsub['subject']=pbestsub['id'].apply(lambda x:x.split('_')[0][4:])
for name in names:
    pbest[name]=0.0
    pbestsub[name]=0.0
import os
X=None
cvfiles=os.listdir('cv/')
for subk in range(1,13):
    X= None
    Xt= None
    for fname in os.listdir('sub/'):
        if fname not in cvfiles or 'sub%d_'%subk not in fname:
            continue
    
        subz=pd.read_csv('sub/'+fname)
        sub2=pd.read_csv('cv/'+fname)
        if X is None:
            X=np.array(sub2[names])
            Xt=np.array(subz[names])
        else:
            X=np.hstack((X,np.array(sub2[names])))
            Xt=np.hstack((Xt,np.array(subz[names])))
    print 'predict sub%d'%subk, X.shape, Xt.shape
    mask=np.array(real['subject']==str(subk))  
    masksub=np.array(pbestsub['subject']==str(subk))
    for name in names:
        
        y=np.array(real[name])
        y=y[mask]
     
        yr=train_predict(X,y,Xt,c=1)
        
        tmp=np.array(pbestsub[name])
        tmp[masksub]=yr
        pbestsub[name]=tmp
    
pbestsub.drop('subject',inplace=True,axis=1)    
pbestsub.to_csv('xgballsub3.csv',index=False)  
