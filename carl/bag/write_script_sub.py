import os
f=open('runsub.sh','w')
subfiles=os.listdir('sub')

for electrode in range(1,33):
    for subject in range(1,13):
        for clf in range(1,7):
            fname='try_sub%d_clf%d_trode%d.csv'%(subject,clf,electrode) 
            if fname in subfiles:
                continue  
            f.write('python trysub.py %d %d %d\n'%(subject,clf,electrode))


f.close()
