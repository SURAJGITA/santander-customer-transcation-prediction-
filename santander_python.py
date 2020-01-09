#!/usr/bin/env python
# coding: utf-8

# In[2]:


# LOADING LIBRARIES
from os import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()

from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import cross_val_score


# In[3]:


# CHECK DIRECTORY AND SET DIRECTORY
getcwd()
chdir('C:\\Users\\My guest\\Desktop\\project 2')
getcwd()


# In[4]:


# read data set
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[5]:


train.shape,test.shape


# In[6]:


train.info(),test.info()


# In[7]:


#droping id_code colum from both the data set
test=test.drop('ID_code',axis=1)
train=train.drop('ID_code',axis=1)


# In[8]:


# changing the data type of target column of train data set
train.target=train.target.astype('category')


# In[9]:


# lets do some visulisation of data
#befor that disciptive analysis
train.describe()


# In[10]:


test.describe()


# In[11]:


train.target.describe()


# In[12]:


# checking the training data for imbalance 
count_classes = pd.value_counts(train.target)
print(count_classes)
count_classes.plot(kind= 'bar',rot=1)


# In[13]:


# Data to plot
labels = 'zeros', 'ones'
sizes = [179902,20098]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# In[14]:


plt.hist(train.head())


# In[15]:


#missing value analysis
train.isna().sum().sum()
test.isna().sum().sum()
# there are no missing values in both the data sets


# In[16]:


train_input=train.drop('target',axis=1) 
columns=train_input.columns
columns


# In[17]:


outliers=0
for i in columns:
    q75=np.percentile(train[i],75)
    q25=np.percentile(train[i],25)
    iqr=q75-q25
    minimum=q25-(iqr*1.5)
    maximum=q75+(iqr*1.5)
    print('q75=',q75,'q25=',q25,'iqr=',iqr,'minimum=',minimum,'maximum=',maximum)
    outlier=(train[i]>maximum).sum()+(train[i]<minimum).sum()
    print('no. of outlier in ',i,'is',outlier)
    outliers=outliers+outlier


# In[18]:


print('no. of outliers=',outliers)


# In[19]:


for i in columns:
    q75=np.percentile(train[i],75)
    q25=np.percentile(train[i],25)
    iqr=q75-q25
    minimum=q25-(iqr*1.5)
    maximum=q75+(iqr*1.5)
    train[i].loc[train[i]>maximum]=np.nan
    train[i].loc[train[i]<minimum]=np.nan


# In[20]:


#train_input['var_0'].loc[train_input['var_0']>maximum]=np.nan
print('no. of nans made out of outliers=',train.isna().sum().sum())


# In[21]:


#removing nans and hence removing all outliers
train=train.dropna()
train.isna().sum().sum()


# In[22]:


train.shape


# In[23]:


# checking the training data for imbalance 
count_classes = pd.value_counts(train.target)
print(count_classes)
count_classes.plot(kind= 'bar',rot=1)


# In[24]:


# Data to plot
labels = 'zeros', 'ones'
sizes = [157999,17105]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# In[25]:


sns.heatmap(train.corr())
svm=sns.heatmap(train.corr())

figure = svm.get_figure()    
figure.savefig('svm_conf.png', dpi=400)


# In[26]:


f = plt.figure(figsize=(19, 15))
plt.matshow(train.corr(), fignum=f.number)
plt.xticks(range(train.shape[1]), train.columns, fontsize=14, rotation=45)
plt.yticks(range(train.shape[1]), train.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);
plt.savefig('correlation matrix.png')


# In[27]:


plt.matshow(train.corr())
plt.show()
plt.savefig('corr.png')


# In[28]:


rs = np.random.RandomState(0)
corr = train.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[29]:


# putting all the df colname in a list
traincols = list(train.columns)

# exculdig target and index columns
variables = traincols[2:]

# splitting the list every n elements:
n = 10
chunks = [variables[x:x + n] for x in range(0, len(variables), n)]


# In[30]:


# displaying a boxplot every n columns:
for i in chunks:
    plt.show(train.boxplot(column = i, sym='k.', figsize=(10,5)))


# In[31]:


# handling imbalance training  data
#HANDLING IMBALANCED DATA BY OVER SAMPLING
os =  RandomOverSampler(ratio=1)
X_train_res, y_train_res = os.fit_sample(train.drop('target',axis=1),train['target'])
X_train_res.shape,y_train_res.shape
print(X_train_res.shape,y_train_res.shape)
print('Original dataset shape {}'.format(Counter(train['target'])))
print('Resampled dataset shape {}'.format(Counter(y_train_res)))


# #modeling using k fold cross validation function
# 

# In[32]:


# Xgb model
#cross_val_score( xgb.XGBClassifier(),X_train_res, y_train_res,cv=3)


# In[33]:


# 35 minutes


# In[34]:


#decision tree
#cross_val_score(tree.DecisionTreeClassifier(),X_train_res,y_train_res,cv=3)


# In[35]:


#random forest
#cross_val_score(RandomForestClassifier(n_estimators=20),X_train_res,y_train_res,cv=3).mean()


# In[36]:


#naive bayes
#cross_val_score(GaussianNB(),X_train_res,y_train_res,cv=3).mean()  


# In[37]:


#logistic regression
#cross_val_score(LogisticRegression(),X_train_res,y_train_res,cv=3)


# In[38]:


#sgd
#cross_val_score(SGDClassifier(),X_train_res,y_train_res,cv=3)


# In[39]:


#from lightgbm import LGBMClassifier
#cross_val_score(LGBMClassifier(),X_train_res,y_train_res,cv=3).mean()


# In[41]:


#from sklearn.model_selection import RandomizedSearchCV
# Create the parameter grid based on the results of random search 
#param_grid = {
    #'bootstrap': [True],
   # 'max_depth': [80, 90, 100, 110],
   # 'max_features': [2, 3],
   # 'min_samples_leaf': [3, 4, 5],
   # 'min_samples_split': [8, 10, 12],
   # 'n_estimators': [100, 200, 300, 1000]
#}
# Create a based model
#rf = RandomForestClassifier()
# Instantiate the grid search model
#random_search = RandomizedSearchCV(estimator = rf, param_distributions= param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
#rf_random=random_search.fit(X_train_res,y_train_res)


# In[42]:


# Fit the random search model
#rf_random.best_params_


# In[43]:


# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_train_res,y_train_res,test_size=0.3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# # model implementation usin random forest

# In[44]:


model_rf = RandomForestClassifier(n_estimators=20)


# In[45]:


model_rf.fit(X_train , y_train)


# In[46]:


prediction=model_rf.predict(X_test)


# In[47]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test,prediction)
CM =pd.crosstab(y_test,prediction)
CM


# In[48]:


from sklearn.metrics import accuracy_score 
accuracy_score(y_test,prediction)*100


# In[49]:


TN=CM.iloc[0,0]
FN=CM.iloc[1,0]
TP=CM.iloc[1,1]
FP=CM.iloc[1,0]


# In[50]:


#false nagative rate
FN*100/(FN+TP)


# In[51]:


# FALSE POSITIVE RATE
FP*100/(TN+FP)


# In[52]:


#RECALL
TP*100/(TP+FN)


# In[53]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test,prediction)
roc_auc = auc(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig('auc_rf')


# 

# In[54]:


prediction=model.predict(test)


# In[55]:


prediction.shape


# In[56]:


test['target']=pd.DataFrame(prediction)


# In[57]:


# checking the training data for imbalance 
count_classes = pd.value_counts(test.target)
print(count_classes)
count_classes.plot(kind= 'bar',rot=1)


# # implementation using NAives bayes

# In[58]:


model=GaussianNB()


# In[59]:


model.fit(X_train , y_train)


# In[60]:


prediction=model.predict(X_test)


# In[61]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test,prediction)
CM =pd.crosstab(y_test,prediction)
CM


# In[62]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test,prediction)
roc_auc = auc(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig('auc_nb')


# In[63]:


test.drop('target',axis=1,inplace=True)
prediction=model.predict(test)


# In[64]:


test['target']=pd.DataFrame(prediction)


# In[65]:


# checking the training data for imbalance 
count_classes = pd.value_counts(test.target)
print(count_classes)
count_classes.plot(kind= 'bar',rot=1)


# # implementation using light gbm

# In[66]:


model=LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=1000, n_jobs=-1, num_leaves=31, objective=None,
               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)


# In[67]:


model.fit(X_train , y_train)


# In[68]:


prediction=model.predict(X_test)


# In[69]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test,prediction)
CM =pd.crosstab(y_test,prediction)
CM


# In[70]:


from sklearn.metrics import accuracy_score 
accuracy_score(y_test,prediction)*100


# In[71]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test,prediction)
roc_auc = auc(fpr,tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig('lightgmb_auc')


# In[72]:


test.drop('target',axis=1,inplace=True)
prediction=model.predict(test)


# In[73]:


test['target']=pd.DataFrame(prediction)


# In[74]:


# checking the training data for imbalance 
count_classes = pd.value_counts(test.target)
print(count_classes)
count_classes.plot(kind= 'bar',rot=1)


# # predictionn the target value for test data set value using 

# In[ ]:


prediction=model_rf.predict(test)
prediction.shape


# In[ ]:


test['target']=pd.DataFrame(prediction)

