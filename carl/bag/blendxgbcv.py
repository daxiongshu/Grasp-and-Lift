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

pbest=pd.read_csv('../vali-stack_srk1/vali1_3_new_cv.csv')
for name in names:
    pbest[name]=0.0
import os
X=None
for subk in range(1,13):
    X= None
    for fname in os.listdir('cv'):
        if 'sub%d.csv'%subk not in fname:
            continue
    
        sub2=pd.read_csv('cv/'+fname)
        if X is None:
            X=np.array(sub2[names])
        else:
            X=np.hstack((X,np.array(sub2[names])))
    # column stack all the models, X.shape[1]=num_electrode * num_classifier * num_events

    print 'predict sub%d'%subk, X.shape
    mask=np.array(real['subject']==str(subk))  
    X1=X[:X.shape[0]/2]
    X2=X[X.shape[0]/2:] 
    for name in names:
        
        y=np.array(real[name])
        y=y[mask]
        
        y1=y[:X.shape[0]/2]
        y2=y[X.shape[0]/2:]
        yr2=train_predict(X1,y1,X2,yt=y2,c=1)
                
        xx.append(myauc(y2,yr2))
        print name, xx[-1]
        yr1=train_predict(X2,y2,X1,yt=y1,c=1)
        xx.append(myauc(y1,yr1))
        print name, xx[-1]
        tmp=np.array(pbest[name])
        tmp[mask]=np.concatenate((yr1,yr2))
        pbest[name]=tmp
    
    print '%d average'%subk,np.mean(xx)
pbest.to_csv('xgball.csv',index=False)  
