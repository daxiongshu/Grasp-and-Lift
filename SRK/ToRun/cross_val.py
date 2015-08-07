# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:40:37 CEST 2015

@author: Elena Cuoco
simple starting script, without the use of MNE
Thanks to @author: alexandrebarachant for his wornderful starting script


"""

import numpy as np
import pandas as pd

from scipy.signal import butter, lfilter, boxcar
from numpy import convolve

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from glob import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.metrics import roc_auc_score

from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.svm import SVC
from sklearn import ensemble

#from joblib import Parallel, delayed

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

def get_fea(X, rand_num=0):
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
    tmp=np.hstack((X,Xm,Xt,Xu,Xs))
    return tmp


scaler= StandardScaler()
l1 = [i-1 for i in [1,3,5,6,9,16,18,20,25,27,29,32]]
#l1 = [i-1 for i in [2,4,7,10,12,14,20,22,23,26,28,30]]
#l1 = [i-1 for i in [2,5,8,11,13,15,17,19,21,24,30,31]]
#for j in [0,1,2,4,6,11,13,15,16,21,23,25,27,29,31]:
#for j in [3,5,7,9,12,14,18,20,22,24,26,28,30]:
#l1 = range(32)
#l1 = [0]
for j in l1:
	for i in range(10):
		exec("scaler"+str(j)+"_"+str(i)+"= StandardScaler()")

def data_preprocess_train(X):
    X_prep = scaler.fit_transform(X)
    #X_prep2 = np.diff(X_prep, axis=0)
    #X_prep2 = np.vstack([X_prep[0],X_prep[:-1]])
    #X_prep3 = np.vstack([X_prep[0],X_prep[0],X_prep[:-2]])

    ##for j in [0,1,2,4,6,11,13,15,16,21,23,25,27,29,31]:
    ##for j in [3,5,7,9,12,14,18,20,22,24,26,28,30]:
    for j in l1:
        X_prep_low = np.zeros((np.shape(X_prep)[0],10))
    	for i in range(10):
            X_prep_low[:,i] = butterworth_filter(X[:,j],1,2-(i*0.2),3)
            exec("X_prep_low[:,i] = scaler"+str(j)+"_"+str(i)+".fit_transform(X_prep_low[:,i])")
        X_prep_low_pow = X_prep_low ** 2
        X_prep = np.concatenate((X_prep_low,X_prep,X_prep_low_pow),axis=1)

    #X_prep = np.concatenate((X_prep, X_prep2, X_prep3), axis=1)

    #X_prep = np.concatenate((X_prep_low,X_prep_normal,X_prep_low_pow,X_prep_low1,X_prep_low_pow1, X_prep_low2,X_prep_low_pow2),axis=1)
    return get_fea( X_prep)

def data_preprocess_test(X):
    X_prep = scaler.transform(X)
    #X_prep2 = np.vstack([X_prep[0],X_prep[:-1]])
    #X_prep3 = np.vstack([X_prep[0],X_prep[0],X_prep[:-2]])

    ##for j in [0,1,2,4,6,11,13,15,16,21,23,25,27,29,31]:
    ##for j in [3,5,7,9,12,14,18,20,22,24,26,28,30]:
    for j in l1:
        X_prep_low = np.zeros((np.shape(X_prep)[0],10))
        for i in range(10):
            X_prep_low[:,i] = butterworth_filter(X[:,j],1,2-(i*0.2),3)
            exec("X_prep_low[:,i] = scaler"+str(j)+"_"+str(i)+".transform(X_prep_low[:,i])")
        X_prep_low_pow = X_prep_low ** 2
        X_prep = np.concatenate((X_prep_low,X_prep,X_prep_low_pow),axis=1)

    #X_prep = np.concatenate((X_prep, X_prep2, X_prep3), axis=1)

    return get_fea( X_prep )


def fit(X,y):
    # Do here you training
    #clf = LogisticRegression(penalty="l2")
    #clf = SVC(kernel='linear', probability=True, random_state=0)
    clf1 = LDA()
    #clf = ensemble.RandomForestClassifier(n_estimators=10, max_depth=8, min_samples_leaf=4, n_jobs=4, random_state=0)
    clf1.fit(X,y)
    #pred_y = clf1.predict_proba(X)[:,[1]]
    #pred_y2 = np.vstack([pred_y[0],pred_y[:-1]])
    #pred_y3 = np.vstack([pred_y[0],pred_y[0],pred_y[:-2]])
    #pred_y = np.concatenate((pred_y, pred_y2, pred_y3),axis=1)
    #clf2 = LDA()
    #clf2.fit(pred_y, y)
    return clf1

def predict(clf,X):
    #clf1, clf2 = clf[0], clf[1]
    # do here your prediction
    preds = clf.predict_proba(X)
    #pred_y2 = np.vstack([pred_y[0],pred_y[:-1]])
    #pred_y3 = np.vstack([pred_y[0],pred_y[0],pred_y[:-2]])
    #pred_y = np.concatenate((pred_y, pred_y2, pred_y3),axis=1)
    #preds = clf2.predict_proba(pred_y)
    return preds[:,1]
    #return np.atleast_2d(preds[:,clf.classes_==1])
    
# training subsample.if you want to downsample the training data
subsample = 79
subsample2 = 98
subsample3 = 61
#series used for CV
series = range(2,9)
#######columns name for labels#############
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']

#######number of subjects###############
subjects = range(1,13)
#subjects = range(1,3)
auc_tot = []
pred_tot = []
y_tot = []
###loop on subjects and 8 series for train data + 2 series for test data
for subject in subjects:
    print "Subject : ", subject
    y_raw= []
    raw = []
    sequence = []
    ################ READ DATA ################################################
    
    for ser in series:
      fname =  '../Data/train/subj%d_series%d_data.csv' % (subject,ser)
      data,labels=prepare_data_train(fname)
      raw.append(data)
      y_raw.append(labels)
      sequence.extend([ser]*len(data))

    X = pd.concat(raw)
    y = pd.concat(y_raw)
    #transform in numpy array
    #transform train data in numpy array
    X = np.asarray(X.astype(float))
    y = np.asarray(y.astype(float))
    sequence = np.asarray(sequence)


    ################ Train classifiers ########################################
    cv = LeaveOneLabelOut(sequence)
    pred = np.zeros((X.shape[0],6))

    for train, test in cv:
	print "cv"
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        #apply preprocessing
        X_train=data_preprocess_train(X_train)
        X_test=data_preprocess_test(X_test)
	for i in range(6):
		#print "i"
		clf = fit(X_train[::subsample,:],y_train[::subsample,i])
		preds1 = predict(clf, X_test)
		clf = fit(X_train[::subsample2,:],y_train[::subsample2,i])
                preds2 = predict(clf, X_test)
		clf = fit(X_train[::subsample3,:],y_train[::subsample3,i])
                preds3 = predict(clf, X_test)
		pred[test,i] = (preds1 + preds2 + preds3) / 3.0

        #clfs = Parallel(n_jobs=6)(delayed(fit)(X_train[::subsample,:],y_train[::subsample,i]) for i in range(6))
        #preds = Parallel(n_jobs=6)(delayed(predict)(clfs[i],X_test) for i in range(6))
        #pred[test,:] = np.concatenate(preds,axis=1)
    pred_tot.append(pred)
    y_tot.append(y)
    # get AUC
    auc = [roc_auc_score(y[:,i],pred[:,i]) for i in range(6)]     
    auc_tot.append(auc)
    print(auc)

pred_tot = np.concatenate(pred_tot)
y_tot = np.concatenate(y_tot)
global_auc = [roc_auc_score(y_tot[:,i],pred_tot[:,i]) for i in range(6)]

print('Global AUC : %.4f' % np.mean(global_auc))
