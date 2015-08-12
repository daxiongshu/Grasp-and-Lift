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
# stack1cv  auc: 0.931
# if the reinforced base mode is used for stacking


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
# read the predictions of the base model
subname='vali1_3_new_sub.csv'  
sub=pd.read_csv(subname,index_col=0)


# this is the training set for this stack model, which is the validation set for the base model
cvname='vali1_3_new_cv.csv'
cv=pd.read_csv(cvname,index_col=0)  
print 'pred', sub.shape

# read the true labels of the training set
subjects = range(1,13)
real=[]
for subject in subjects:
    fnames =  ['../../../Data/train/subj%d_series%d_events.csv' % (subject,i) for i in range(7,9)]
    for fname in fnames:
        labels= pd.read_csv(fname,index_col=0)
        real.append(labels)
        print fname,labels.shape
real=pd.concat(real)
print 'combined', real.shape
print "column names ", real.columns.values

# generating features based on past predictions.
def gendata(X):
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
            Xu[i,:]=np.mean(X[i-m1:i,:],axis=0)
        elif i<m3:
            Xm[i,:]=np.mean(X[i-m2:i-m1,:],axis=0)
            Xu[i,:]=np.mean(X[i-m1:i,:],axis=0)

        elif i<m4:
            Xm[i,:]=np.mean(X[i-m2:i-m1,:],axis=0)
            Xu[i,:]=np.mean(X[i-m1:i,:],axis=0)
            Xt[i,:]=np.mean(X[i-m3:i-m2,:],axis=0)
        else:
            Xm[i,:]=np.mean(X[i-m2:i-m1,:],axis=0)
            Xu[i,:]=np.mean(X[i-m1:i,:],axis=0)
            Xt[i,:]=np.mean(X[i-m3:i-m2,:],axis=0)
            Xs[i,:]=np.mean(X[i-m4:i-m3,:],axis=0)
    return np.hstack((X,Xm,Xt,Xu,Xs))

# the data needed is already generated in the cv process.
X=np.array(cv[real.columns.values]) 
X=gendata(X)


#h5f=h5py.File('h5file/stack1cv.h5','r')
#X=h5f['dataset_1'][:]
#h5f.close()

# we need to generate the features for the test set
# this only need to be done once.
Xt=np.array(sub[real.columns.values])
Xt=gendata(Xt)

#h5f=h5py.File('h5file/vali1sub.h5','w')
#h5f.create_dataset('dataset_1', data=Xt)
#h5f.close()

#next time we can simple load the data
#h5f=h5py.File('h5file/vali1sub.h5','r')
#Xt=h5f['dataset_1'][:]
#h5f.close()

print "Shape of train, train_y, test", X.shape, real.shape, Xt.shape
xx=[]
subx=sub.copy()
for name in real.columns.values:
    y=np.array(real[name])
    subx[name] =train_predict(X,y,Xt,c=1)
    print name,'done'

subx.to_csv('stack1_3_new_sub.csv')
