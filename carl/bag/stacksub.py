from xgb_classifier import xgb_classifier
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from collections import Counter
from sklearn import metrics
import h5py


def train_predict(X,y,Xt,yt=[],c=1):
    if c==1:
        #clf=xgb_classifier(num_round=45,eta=0.1,min_child_weight=5,depth=10, subsample=0.5,col=1) 
        clf=xgb_classifier(num_round=45,eta=0.1,min_child_weight=20,depth=20, subsample=0.1,col=0.7)
        return clf.train_predict(X,y,Xt,yt)

import pickle

    #pickle.dump(rf,open('yxgbc_fea1.p','w'))

def myauc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)

import sys


subname='../vali-stack8/vali8_7sub.csv'  # this has future information
sub=pd.read_csv(subname,index_col=0)

subname='../vali-stack3/vali3_1_sub.csv' 
sub1=pd.read_csv(subname,index_col=0)

subname='../nn3/subm.csv'  
sub2=pd.read_csv(subname,index_col=0)

subname='xgballsub4.csv'  
sub3=pd.read_csv(subname,index_col=0)

print 'pred', sub.shape
subjects = range(1,13)
real=[]
for subject in subjects:
    fnames =  ['../../data/train/subj%d_series%d_events.csv' % (subject,i) for i in range(7,9)]
    for fname in fnames:
        labels= pd.read_csv(fname,index_col=0)
        real.append(labels)
        print fname,labels.shape
real=pd.concat(real)
print 'combined', real.shape

def gendata(X,Xp):
    m1,m2,m3,m4=150,300,600,1000
    Xm=np.zeros(X.shape)
    Xs=np.zeros(X.shape)
    Xt=np.zeros(X.shape)
    Xu=np.zeros(X.shape)
    Xv=np.zeros(X.shape)
    for i in range(X.shape[0]):
        if i<m1:
            pass
        elif i<m2:
            Xu[i,:]=np.mean(Xp[i-m1:i,:],axis=0)
        elif i<m3:
            Xm[i,:]=np.mean(X[i-m2:i-m1,:],axis=0)
            Xu[i,:]=np.mean(Xp[i-m1:i,:],axis=0)

        elif i<m4:
            Xm[i,:]=np.mean(X[i-m2:i-m1,:],axis=0)
            Xu[i,:]=np.mean(Xp[i-m1:i,:],axis=0)
            Xt[i,:]=np.mean(X[i-m3:i-m2,:],axis=0)
        else:
            Xm[i,:]=np.mean(X[i-m2:i-m1,:],axis=0)
            Xu[i,:]=np.mean(Xp[i-m1:i,:],axis=0)
            Xt[i,:]=np.mean(X[i-m3:i-m2,:],axis=0)
            Xs[i,:]=np.mean(X[i-m4:i-m3,:],axis=0)
    return np.hstack((Xp,Xm,Xt,Xu,Xs))





#h5f=h5py.File('h5file/stack1cv.h5','w')
#h5f.create_dataset('dataset_1', data=X)
#h5f.close()
h5f=h5py.File('h5file/stacksrk1cv1.h5','r')
X0=h5f['dataset_1'][:]
h5f.close()

h5f=h5py.File('../nn3/h5file/stacknn3.h5','r')
X=h5f['dataset_1'][:]
h5f.close()

h5f=h5py.File('../vali-stack8/h5file/stack8cv.h5','r')
tmp=h5f['dataset_1'][:]
h5f.close()

X=np.hstack((X0,X,tmp))

Xt=np.array(sub[real.columns.values])
Xp=np.array(sub1[real.columns.values])
tmp=gendata(Xt,Xp)

Xt=np.array(sub2[real.columns.values])
Xt=gendata(Xt,Xt)

Xt0=np.array(sub3[real.columns.values])
Xt0=gendata(Xt0,Xt0)

Xt=np.hstack((Xt0,Xt,tmp))

print 'done',X.shape,Xt.shape
xx=[]
subx=sub.copy()
for name in real.columns.values:
    y=np.array(real[name])
 
    yr=train_predict(X,y,Xt,c=1)
    
   
    subx[name]=yr
    

subx.to_csv('stackxgballsub5.csv')

