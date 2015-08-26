f=open('runcv.sh','w')
for electrode in [i for i in range(2,33) if i not in [1,3,25,32]]:
    for subject in range(1,13):
        for clf in range(1,7):
            f.write('python trycv.py %d %d %d\n'%(subject,clf,electrode))


f.close()
