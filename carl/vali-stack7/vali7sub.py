# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:40:37 CEST 2015

@author: Elena Cuoco
simple starting script, without the use of MNE
Thanks to @author: alexandrebarachant for his wornderful starting script
average 0.816735586598
average 0.835100413117
average 0.854357544601
average 0.904356722514
average 0.911726630906
"""
import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, boxcar
from numpy import convolve
from sklearn.linear_model import LogisticRegression
from glob import glob
import os
from sklearn.lda import LDA
from sklearn.preprocessing import StandardScaler
from sklearn.qda import QDA
from sklearn.ensemble import RandomForestClassifier
 
#############function to read data###########

def prepare_data_train(fname):
    """ read and prepare training data """
    # Read data
    data = pd.read_csv(fname)
    # events file
    events_fname = fname.replace('_data','_events')
    # read event file
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)#remove id
    labels=labels.drop(['id' ], axis=1)#remove id
    return  clean,labels

def butterworth_filter(X,t,k,l):
    if t==0:
        freq=[k, l]
        b,a = butter(3,np.array(freq)/500.0,btype='bandpass')
        X = lfilter(b,a,X)
    elif t==1:
        b,a = butter(3,k/500.0,btype='lowpass')
        X = lfilter(b,a,X)
    elif t==2:
        b,a = butter(3,l/500.0,btype='highpass')
        X = lfilter(b,a,X)      
    return X

def prepare_data_test(fname):
    """ read and prepare test data """
    # Read data
    data = pd.read_csv(fname)
    return data

scaler= StandardScaler()
def data_preprocess_train(X,sub=-1):
    X_prep_normal = scaler.fit_transform(X)
    X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
    for i in range(10):
        X_prep_low[:,i] = butterworth_filter(X[:,0],1,2-(i*0.2),3)
        X_prep_low[:,i] = scaler.fit_transform(X_prep_low[:,i])
    X_prep_low_pow = X_prep_low ** 2
    X_prep = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)
    #do here your preprocessing
    
    return get_fea(X_prep,cat='train',sub=sub)

def data_preprocess_test(X,sub=-1):
    X_prep_normal = scaler.fit_transform(X)
    X_prep_low = np.zeros((np.shape(X_prep_normal)[0],10))
    for i in range(10):
        X_prep_low[:,i] = butterworth_filter(X[:,0],1,2-(i*0.2),3)
        X_prep_low[:,i] = scaler.fit_transform(X_prep_low[:,i])
    X_prep_low_pow = X_prep_low ** 2
    X_prep = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow),axis=1)
    return get_fea(X_prep,cat='test',sub=sub)

# training subsample.if you want to downsample the training data
subsample  = 70
subsample2 = 130
#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']




#######number of subjects###############
subjects = range(1,13)
ids_tot = []
pred_tot = []

###loop on subjects and 8 series for train data + 2 series for test data

def get_fea(X,cat,sub):
    m1,m2=75,150
    Xm=np.zeros(X.shape)
    Xs=np.zeros(X.shape)
    
    for i in range(X.shape[0]):
        if i<X.shape[0]-m2:
            Xm[i,:]=np.mean(X[i+m1:i+m2,:],axis=0)
            Xs[i,:]=np.mean(X[i:i+m1,:],axis=0)
        elif i<X.shape[0]-m1:
            Xs[i,:]=np.mean(X[i:i+m1,:],axis=0)
        
    tmp=np.hstack((Xm,Xs))
    h5f=h5py.File('h5file/%s_%d.h5'%(cat,sub),'w')
    h5f.create_dataset('dataset_1', data=tmp)
    h5f.close()
    h5f=h5py.File('../true/h5sub/%s_%d.h5'%(cat,sub),'r')
    X=h5f['dataset_1'][:]
    h5f.close()
    print X.shape, tmp.shape
    return np.hstack((tmp,X))
for subject in subjects:
    y_raw= []
    raw = []
    ################ READ DATA ################################################
    fnames =  ['../../data/train/subj%d_series%d_data.csv' % (subject,i) for i in range(1,9)]
    #print fnames[:-2]
    #assert(False)
    for fname in fnames:
      data,labels=prepare_data_train(fname)
      raw.append(data)
      y_raw.append(labels)

    X = pd.concat(raw)
    y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
    X_train =np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))


    ################ Read test data #####################################
    #
    fnames =  ['../../data/test/subj%d_series%d_data.csv' % (subject,i) for i in range(9,11)]
    test = []
    idx=[]
    for fname in fnames:
      data=prepare_data_test(fname)
      test.append(data)
      idx.append(np.array(data['id']))
    X_test= pd.concat(test)
    ids=np.concatenate(idx)
    ids_tot.append(ids)
    X_test=X_test.drop(['id' ], axis=1)#remove id
    #transform test data in numpy array
    X_test =np.asarray(X_test.astype(float))


    ################ Train classifiers ########################################
    lr1 = LDA()#RandomForestClassifier(n_jobs=16,n_estimators=100)
    lr2=LogisticRegression()
    lr3 = LDA()
   
    
    pred1 = np.empty((X_test.shape[0],6))
    pred2 = np.empty((X_test.shape[0],6))
    pred3 = np.empty((X_test.shape[0],6))
    

    pred = np.empty((X_test.shape[0],6))
    
    X_train=data_preprocess_train(X_train,sub=subject)
    X_test=data_preprocess_test(X_test,sub=subject)
    for i in range(6):
        y_train= y[:,i]
        print('Train subject %d, class %s' % (subject, cols[i]))
        #lr1.fit(X_train[::subsample,:],y_train[::subsample])
        lr2.fit(X_train[::subsample,:],y_train[::subsample])
        #lr3.fit(X_train[::subsample2,:],y_train[::subsample2])
        
        #pred1[:,i] = lr1.predict_proba(X_test)[:,1]
        pred[:,i] = lr2.predict_proba(X_test)[:,1]
        #pred3[:,i] = lr3.predict_proba(X_test)[:,1]
        
        #pred[:,i]=pred1[:,i]*0.24+pred2[:,i]*0.24+pred3[:,i]*0.52
    pred[:300,:]=0
    pred_tot.append(pred)

# submission file
submission_file = 'vali7sub.csv'
# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')
