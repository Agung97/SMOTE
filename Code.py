# import modul yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import missingno as msno 
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

#Cek rasio label data
df['test_outcome'].value_counts()/df.shape[0]

#Split data dengan metode stratified
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, stratify=df.test_outcome)
X_train = train.drop(['test_outcome'], axis=1)
y_train = train['test_outcome']
X_test = test.drop(['test_outcome'], axis=1)
y_test = test['test_outcome']
train.pivot_table(index='test_outcome', aggfunc='size').plot(kind='bar', title='Verify that class distribution in train is same as input data')
test.pivot_table(index='test_outcome', aggfunc='size').plot(kind='bar', title='Verify that class distribution in train is same as input data')

#----------------------------------------------------------------------------------------------------------------------------------------#
# Oversampling pada data train
print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

# import SMOTE module from imblearn library
# pip install imblearn (install modul imblearn pada command jika belum terinstall)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 2)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

#------------------------------------------------------------------------------------------------------------------------------------------------#
#Oversampling pada data testing
X_test_res, y_test_res = smote.fit_resample(X_test, y_test.ravel())
print("Before OverSampling, counts of label '1': {}".format(sum(y_test == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_test == 0)))

print('After OverSampling, the shape of test_X: {}'.format(X_test_res.shape))
print('After OverSampling, the shape of test_y: {} \n'.format(y_test_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_test_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_test_res == 0)))

#---------------------------------------------------------------------------------------------------------------------------------------------------#
#Mengecek distribusi label pada data setelah Oversampling
#Data training
pd.Series(y_train_res).value_counts().plot(kind='bar', title='Class distribution after applying SMOTE', xlabel='test_outcome')

#Data testing
pd.Series(y_test_res).value_counts().plot(kind='bar', title='Class distribution after applying SMOTE', xlabel='test_outcome')

#End
