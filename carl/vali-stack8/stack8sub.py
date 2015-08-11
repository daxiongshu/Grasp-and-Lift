from xgb_classifier import xgb_classifier
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from collections import Counter
from sklearn import metrics
import h5py

########################################
# this stacking uses the base model vali1_cv.csv
# which uses past feature only
# vali1_cv auc: 0.911
# after stacking
# stack1_cv  auc: 0.931
# 

########################################
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
#subname='../btb/cv13_15_19_smooth.csv'
subname='vali8_7.csv'  # this has future information
cv=pd.read_csv(subname,index_col=0)

subname='../vali-stack3/vali3_1.csv'  # this doesn't have future information
cv1=pd.read_csv(subname,index_col=0)

subname='vali8_7sub.csv'  # this has future information
sub=pd.read_csv(subname,index_col=0)

subname='../vali-stack3/vali3_1_sub.csv'  # this doesn't have future information
sub1=pd.read_csv(subname,index_col=0)

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

#next time we run it, just load the data
h5f=h5py.File('h5file/stack8cv.h5','r')
X=h5f['dataset_1'][:]
h5f.close()

Xt=np.array(sub[real.columns.values])
Xp=np.array(sub1[real.columns.values])
Xt=gendata(Xt,Xp)
print 'done',X.shape
xx=[]
subx=sub.copy()
for name in real.columns.values:
    y=np.array(real[name])
 
    yr=train_predict(X,y,Xt,c=1)
    
   
    subx[name]=yr
    

subx.to_csv('stack8sub.csv')

