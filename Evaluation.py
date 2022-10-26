#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the models
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AER_credit_card_data.csv'


# In[4]:


df = pd.read_csv(data)


# In[6]:


df.head()


# In[8]:


df.describe()


# In[10]:


df.shape


# In[13]:


#data preparation: Create the target variable by mapping yes to 1 and no to 0.
card_map = {
    "yes":1,
    "no":0
}
df["card"] = df.card.map(card_map)


# In[14]:


df.head()


# In[15]:


#seperating the numerical amd categorical variables
num = ["reports", "age", "income", "share", "expenditure", "dependents", "months", "majorcards", "active"]
cat = ["owner", "selfemp"]


# In[16]:


#splitting the datasets
from sklearn.model_selection import train_test_split


# In[18]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)
len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

y_train = df_train.card
y_val = df_val.card
y_test = df_test.card

del df_train['card']
del df_val['card']
del df_test['card']


# In[25]:


len(df_train), len(df_val), len(df_test)


# In[26]:


#ROC curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


# In[29]:


for c in num:
    auc = roc_auc_score(y_train, df_train[c])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[c])
    print("%9s, %.3f" % (c, auc))


# In[30]:


#answer to question 1 = Share


# In[31]:


#Training the model
columns = cat + num

train_dicts = df_train[columns].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

val_dicts = df_val[columns].to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]


# In[33]:


#What's the AUC of this model on the validation dataset? (round to 3 digits)
roc_auc_score(y_val, y_pred)


# In[34]:


#answer = 0.995


# In[36]:


#computing precision and recall
def tpr_fpr_dataFrame(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)
    
    return df_scores


df_scores = tpr_fpr_dataFrame(y_val, y_pred)
df_scores[::10]


# In[47]:


df_scores['p'] = df_scores.tp / (df_scores.tp + df_scores.fp)
df_scores['r'] = df_scores.tp / (df_scores.tp + df_scores.fn)


# In[48]:


#plotting the graph
plt.plot(df_scores.threshold, df_scores['p'], label='precision')
plt.plot(df_scores.threshold, df_scores['r'], label='recall')

plt.legend()
plt.show()


# In[41]:


#At which threshold precision and recall curves intersect? = 0.3


# In[49]:


#getting F1 score
df_scores['f1'] = 2 * df_scores.p * df_scores.r / (df_scores.p + df_scores.r)


# In[50]:


plt.plot(df_scores.threshold, df_scores.f1)
plt.xticks(np.linspace(0, 1, 11))
plt.show()


# In[51]:


#At which threshold F1 is maximal? = 0.4


# In[52]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[53]:


scores = []

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.card
    y_val = df_val.card

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))


# In[54]:


#How large is standard devidation of the AUC scores across different folds? =0.003


# In[55]:


kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for C in [0.01, 0.1, 1, 10]:
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.card
        y_val = df_val.card

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%4s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[56]:


#Which C leads to the best mean score? = 1


# In[ ]:




