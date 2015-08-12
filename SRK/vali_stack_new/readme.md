##### This folder has the codes that produced the submission stack1_3_new_sub.7z which scored 0.96033 in the public LB

###### Order of files to run to get the CV:
1. vali1_new.py
2. vali3_new.py
3. python weight.py vali1_new_cv.csv vali3_new_cv.csv
4. stack1_3_new_cv.py

###### Order of files to run to get the Submission:
1. vali1_new_sub.py
2. vali3_new_sub.py
3. python weight.py vali1_new_sub.csv vali3_new_sub.csv
4. stack1_3_new_sub.py

**Val score of base model : 0.925**

**Val score of stacked mdoel : 0.9468**

The base model is an ensemble of LDA and LR as opposed to using only LR in our previous versions. 
This gets the scores from electrodes 1 and 3 individually and then stack the combined results.

**I have removed the future leakage in scalers as well as in stacking models. 
Please let me know if you find any other future leakage in this version.** 
