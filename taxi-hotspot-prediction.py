#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read training data
df_train = pd.read_csv('../input/taxihotspotspredict/taxi_data/train_hire_stats.csv')
# df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train.shape


# In[3]:


# holidays 連假的平常日
# makeupworkdays 連假的補班日

holidays= {'2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12','2016-02-29', '2016-04-04', '2016-04-05', '2016-6-9', '2016-6-10', 
'2016-09-15', '2016-09-16', '2016-10-10', '2017-01-02', '2017-01-27', '2017-01-30', '2017-02-01','2017-02-27', '2017-02-28'}
makeupworkdays = {'2016-06-04', '2016-09-10', '2017-02-18'}


# In[4]:


df_train.head()


# In[5]:


# tag weekday & workday

isworkday = np.ones((len(df_train),), dtype=int)
weekday = np.ones((len(df_train),), dtype=int)
Yisworkday = np.ones((len(df_train),), dtype=int)
Tisworkday = np.ones((len(df_train),), dtype=int)


# In[6]:


#Compute weekday & workday
#weekday()
#0 == Monday, #1 == Tuesday, #2 == Wednesday,  #3 == Thursday,  #4 == Friday,  #5 == Saturday, #6 == Sunday

from datetime import datetime

for index, row in df_train.iterrows():
    row['Date'] = row['Date'].replace('/', '-')
    if row['Date'] in holidays:
        isworkday[index] = 0
    else:
        dd=datetime.strptime(row['Date'], "%Y-%m-%d")
        weekday[index]= dd.weekday() 
        if weekday[index] >=5 and row['Date'] not in makeupworkdays:
            isworkday[index] = 0
    if index > 23:
        Yisworkday[index] = Yisworkday[index-24]
    if index < len(df_train)-25:
        Tisworkday[index+24] = isworkday[index]


# In[7]:


#Build a new dataframe from the training data

RawX = pd.DataFrame(df_train[["Zone_ID", "Hour_slot","Hire_count"]])
RawX['isworkday'] = isworkday
RawX['weekday'] = weekday
RawX['Yisworkday'] = Yisworkday
RawX['Tisworkday'] = Tisworkday


# In[8]:


print(RawX.shape)


# In[9]:


y = RawX["Hire_count"].values
y.shape
RawX=RawX.drop(columns=['Hire_count'])


# In[10]:


#See the raw input data

RawX.head()


# In[11]:


#Use OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')


# In[12]:


#Build encoder

enc.fit_transform(RawX)
enc.categories_


# In[13]:


#Transform data into one hot vector

X = enc.transform(RawX).toarray()
X.shape


# In[14]:


#See the cooked input data
X[0:3, :]


# In[15]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[16]:


model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))


# In[17]:


from tensorflow.keras.optimizers import Adam

model.compile(loss='mse', optimizer=Adam(lr=1e-2,decay=1e-5))
model.fit(X, y, epochs=50, batch_size=5000, verbose=1)


# In[18]:


model.summary()


# In[19]:


df_test = pd.read_csv('../input/taxihotspotspredict/taxi_data/test_hire_stats.csv')
df_test.shape


# In[20]:


#Declare weekday & workday

isworkday2 = np.ones((len(df_test),), dtype=int)
weekday2 = np.ones((len(df_test),), dtype=int)
Yisworkday2 = np.ones((len(df_test),), dtype=int)
Tisworkday2 = np.ones((len(df_test),), dtype=int)


# In[21]:


for index, row in df_test.iterrows():
    if row['Date'] in holidays:
        isworkday2[index] = 0
    else:
        dd=datetime.strptime(row['Date'], "%Y-%m-%d")
        weekday2[index]= dd.weekday() 
        if weekday2[index] >=5 and row['Date'] not in makeupworkdays:
            isworkday2[index] = 0
    if index > 23:
        Yisworkday2[index] = Yisworkday2[index-24]
    if index < len(df_test)-25:
        Tisworkday2[index+24] = isworkday2[index]


# In[22]:


Test = pd.DataFrame(df_test[["Zone_ID", "Hour_slot"]])
Test['isworkday'] = isworkday2
Test['weekday'] = weekday2
Test['Yisworkday'] = Yisworkday2
Test['Tisworkday'] = Tisworkday2


# In[23]:


Xtest = enc.transform(Test).toarray()
Xtest.shape


# In[24]:


yt = model.predict(Xtest)


# In[25]:


yt = yt.astype(int)


# In[26]:


plt.figure(figsize=(20,5))
plt.xlabel('Index',fontsize=20)
plt.ylabel('Hire_count',fontsize=20)
plt.plot(yt)


# In[27]:


test_df=pd.read_csv('../input/taxihotspotspredict/taxi_data/test_hire_stats.csv',sep=',')
test_df['Hire_count']=yt
test_df.head()


# In[28]:


test_df['Hire_count']


# In[29]:


test_df.to_csv('predict.csv',index=False)

