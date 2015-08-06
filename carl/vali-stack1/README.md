Description:

vali1.py: this is a validation code of a base model

for each subject, series 1~6 are used as train and series 7,8 are used as validation

all the features used here are from the past of the frame so no future leakage

vali1sub.py: the same model as vali1.py but training on all training data and predict the test data

it generates a submission file vali1sub.csv, which is a legit submission

stack1cv.py: this is a validation code of a stacking model.

note that this model is fundamentally different from a base model.
It is not a per subject model.
Predictions of the series 7,8 of all subjects are contenated to one big matrix as the training data.
Each row and row still maintains the time order except for boundries of series but it doesn't matter.
Then it performs a 2-fold cross validation fashion.
First, use 1st half data to train and predict 2nd half data and vice versa.
We can not use Kfold random split here because we need to maintain the time order.
This stacking only use predictions from the past of the frame so no future leakage.

stack1sub.py: the same model as stack1cv.py but training on all validation data and predict the test data
it generates a submission file stack1sub.csv, which is a legit submission

vali1: cv auc 0.911
stack1: cv auc 0.931 lb auc 0.950
