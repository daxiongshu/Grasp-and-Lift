# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:40:37 CEST 2015

@author: Elena Cuoco
simple starting script, without the use of MNE
Thanks to @author: alexandrebarachant for his wornderful starting script


"""

from __future__ import division
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
import h5py
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

############# Five different sets of electrodes ##################
l1 = [i-1 for i in [3]]#16,18,20,25,27,29,32]]
#l1 = [i-1 for i in [2,4,7,10,12,14,20,22,23,26,28,30]]
#l1 = [i-1 for i in [2,5,8,11,13,15,17,19,21,24,30,31]]
#l1 = [0,1,2,4,6,11,13,15,16,21,23,25,27,29,31]
#l1 = [3,5,7,9,12,14,18,20,22,24,26,28,30]

for j in l1:
    for i in range(10):
        exec("scaler"+str(j)+"_"+str(i)+"= StandardScaler()")

def data_preprocess_train(X,subject):
    X_prep = scaler.fit_transform(X)

    for j in l1:
        X_prep_low = np.zeros((np.shape(X_prep)[0],10))
        for i in range(10):
            X_prep_low[:,i] = butterworth_filter(X[:,j],1,2-(i*0.2),3)
            exec("X_prep_low[:,i] = scaler"+str(j)+"_"+str(i)+".fit_transform(X_prep_low[:,i])")
        X_prep_low_pow = X_prep_low ** 2
        X_prep = np.concatenate((X_prep_low,X_prep,X_prep_low_pow),axis=1)

    
    return get_fea(X_prep,cat='traincv',sub=subject)

def data_preprocess_test(X,subject):
    X_prep = scaler.transform(X)

    for j in l1:
        X_prep_low = np.zeros((np.shape(X_prep)[0],10))
        for i in range(10):
            X_prep_low[:,i] = butterworth_filter(X[:,j],1,2-(i*0.2),3)
            exec("X_prep_low[:,i] = scaler"+str(j)+"_"+str(i)+".transform(X_prep_low[:,i])")
        X_prep_low_pow = X_prep_low ** 2
        X_prep = np.concatenate((X_prep_low,X_prep,X_prep_low_pow),axis=1)

    return get_fea(X_prep,cat='testcv',sub=subject)


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
    h5f=h5py.File('../vali-stack3/h5file/%s_%d.h5'%(cat,sub),'r')
    X=h5f['dataset_1'][:]
    h5f.close()
    return np.hstack((tmp,X))

# training subsample.if you want to downsample the training data
subsample  = 79
subsample2 = 98
subsample3 = 61
#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']




#######number of subjects###############
subjects = range(1,13)
ids_tot = []
pred_tot = []

###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    print "Subject is ", subject
    y_raw= []
    raw = []
    ################ READ DATA ################################################
    print "Reading train.."
    fnames =  ['../../data/train/subj%d_series%d_data.csv' % (subject,i) for i in range(1,7)]
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
    del X
    del raw
    import gc
    gc.collect()

    ################ Read test data #####################################
    print "Reading test.."
    fnames =  ['../../data/train/subj%d_series%d_data.csv' % (subject,i) for i in range(7,9)]
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
    del test


    ################ Train classifiers ########################################
    lr1 = LogisticRegression()
    lr2 = LogisticRegression()
    #lr2=LogisticRegression(penalty='l1', C=2.0)
    lr3 = LogisticRegression()
    #lr4 = LDA()
    #lr5= LogisticRegression(penalty='l1', C=0.5)
    #lr6 = LogisticRegression(penalty='l1', C=0.5)
    
    #pred1 = np.empty((X_test.shape[0],6))
    #pred2 = np.empty((X_test.shape[0],6))
    #pred3 = np.empty((X_test.shape[0],6))
    #pred4 = np.empty((X_test.shape[0],6))
    #pred5 = np.empty((X_test.shape[0],6))
    #pred6 = np.empty((X_test.shape[0],6))

    pred = np.empty((X_test.shape[0],6))
   
    X_train=data_preprocess_train(X_train,subject)
    X_test=data_preprocess_test(X_test,subject)
    for i in range(6):
        y_train= y[:,i]
        print('Train subject %d, class %s' % (subject, cols[i]))
        lr1.fit(X_train[::subsample,:],y_train[::subsample])
        #lr2.fit(X_train[::subsample2,:],y_train[::subsample2])
        #lr3.fit(X_train[::subsample3,:],y_train[::subsample3])
	#lr4.fit(X_train[::subsample3,:],y_train[::subsample3])
	#lr5.fit(X_train[::subsample2,:],y_train[::subsample2])
        #lr6.fit(X_train[::subsample3,:],y_train[::subsample3])
        pred[:,i] = lr1.predict_proba(X_test)[:,1]
        #pred2[:,i] = lr2.predict_proba(X_test)[:,1]
        #pred3[:,i] = lr3.predict_proba(X_test)[:,1]
	#pred4[:,i] = lr4.predict_proba(X_test)[:,1]
	#pred5[:,i] = lr5.predict_proba(X_test)[:,1]
        #pred6[:,i] = lr6.predict_proba(X_test)[:,1]
        #pred[:,i]=(pred1[:,i]+pred2[:,i]+pred3[:,i])/3.0 #+pred4[:,i]+pred5[:,i]+pred6[:,i])/6.0

    pred_tot.append(pred)

# submission file
submission_file = 'vali8.csv'
# create pandas object for sbmission
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))

# write file
submission.to_csv(submission_file,index_label='id',float_format='%.3f')
