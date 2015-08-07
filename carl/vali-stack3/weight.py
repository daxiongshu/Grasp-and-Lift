import numpy as np
import pandas as pd
from sklearn import metrics
from glob import glob
import sys
subname1=sys.argv[1]
subname2=sys.argv[2]
def myauc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)

sub1=pd.read_csv(subname1,index_col=0)
sub2=pd.read_csv(subname2,index_col=0)
print 'pred', sub1.shape
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

yy=[6, 6, 6, 6, 7, 6]  # vali3, vali1
weight={}
for c,name in enumerate(real.columns.values):
    weight[name]=yy[c]
xx=[]
print weight
for name in real.columns.values:
    
    y=np.array(real[name])
    yr1=np.array(sub1[name])
    yr2=np.array(sub2[name])

    yr=yr1*weight[name]+yr2*(10-weight[name])
    yr/=10
    #print name,j,myauc(y,yr)
    sub1[name]=yr
print weight
#sub1.to_csv('vali3_1.csv')
sub1.to_csv('vali3_1_sub.csv')
