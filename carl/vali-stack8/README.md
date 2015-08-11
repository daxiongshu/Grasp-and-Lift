# This is a correct version of reinforced stacking.
vali8.py generating prediction for the validation set, series 7 and 8 for each subject.

vali8sub.py generating prediction for the test set.

vali8 uses future features, so vali8sub itself is not a legit model

vali8 uses channel 2 for filtering

note that vali7 also uses future features but it is correct, the error occurs in stack7

stack8 fixed the problem of stack7. Specifically,  line 84 in stack8sub.py

return np.hstack((Xp,Xm,Xt,Xu,Xs) # correct
return np.hstack((X,Xm,Xt,Xu,Xs) # wrong
