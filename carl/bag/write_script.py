f=open('runcv.sh','w')
for subject in range(1,13):
    for clf in range(1,7):
        for electrode in [1,3,25,32]
            f.write('python trycv.py %d %d %d\n'%(subject,clf,electrode))


f.close()
