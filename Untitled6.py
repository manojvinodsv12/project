#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv('adult.csv')


# In[2]:


data


# In[3]:


data.columns= data.columns.str.replace(' ','',regex=True)


# In[4]:


data.workclass[data.workclass==" ?"] =np.nan


# In[5]:


data.occupation[data.occupation==" ?"] =np.nan


# In[6]:


data['Native-country'][data['Native-country']==" ?"] =np.nan


# In[7]:


data["Native-country"].value_counts()


# In[8]:


data["country"]="Non-US"
data["country"][data["Native-country"]==" United-States"]="US"


# In[9]:


data


# In[10]:


data["income"].value_counts()
    


# In[11]:


data['income_flag'] = data['income'].map({' <=50K':0, ' >50K':1})


# In[12]:


data


# In[13]:


data['marital-status'].value_counts()


# In[14]:


data["marital"]="married"
data["marital"][data["marital-status"]==" Never-married"]="single"


# In[15]:


data


# In[16]:


data['workclass'].value_counts()


# In[17]:


data["workcl"]="private"
data["workcl"][data["workclass"]==" Local-gov"]="government"
data["workcl"][data["workclass"]==" State-gov"]="government"
data["workcl"][data["workclass"]==" Federal-gov"]="government"


# In[18]:


data


# In[19]:


data['occupation'].value_counts()


# In[20]:


data["occ"]="service"
data["occ"][data["occupation"]==" Protective-serv"]="non service"
data["occ"][data["occupation"]==" Priv-house-serv"]="non service"
data["occ"][data["occupation"]==" Handlers-cleaners"]="non service"
data["occ"][data["occupation"]==" Transport-moving"]="non service"
data["occ"][data["occupation"]==" Other-service"]="non service"
data["occ"][data["occupation"]==" Craft-repair"]="non service"
data["occ"][data["occupation"]==" Armed-Forces"]=" Armed-Forces"
data["occ"][data["occupation"]==" Machine-op-inspct"]="non service"


# In[21]:


data['race'].value_counts()


# In[22]:


data


# In[ ]:





# In[23]:


data['education'].value_counts()


# In[24]:


data["educt"]="school Grade"
data["educt"][data["education"]==" Some-college"]="graduate"
data["educt"][data["education"]==" Masters"]="post graduate"
data["educt"][data["education"]==" Bachelors"]="graduate"
data["educt"][data["education"]==" Doctorate"]="post graduate"


# In[25]:


data


# In[26]:


data.groupby(['occ','workcl','income_flag','educt']).size().unstack()


# In[27]:


data['educt'].value_counts()


# In[28]:


data.groupby(['income_flag','educt']).size().unstack()


# In[29]:


data.info()


# In[36]:


x = data[['sex','marital','occ','workcl','educt','race','country','Hours-per-week','Capital-loss','capital_gain']]
y = data['income_flag']


# In[37]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()


# In[38]:


data['sex']=label.fit_transform(data['sex'])
data['occ']=label.fit_transform(data['occ'])
data['occ'] =pd.get_dummies(data['occ'])
data['workcl']=label.fit_transform(data['workcl'])
data['workcl'] =pd.get_dummies(data['workcl'])
data['educt']=label.fit_transform(data['educt'])
data['educt'] =pd.get_dummies(data['educt'])
data['marital']=label.fit_transform(data['marital'])
data['race']=label.fit_transform(data['race'])
data['race'] =pd.get_dummies(data['race'])
data['country']= label.fit_transform(data['country'])


# In[39]:


data


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)


# In[41]:


from sklearn.linear_model import LogisticRegression
s = LogisticRegression()
s.fit(x_train,y_train)


# In[42]:


s.score(x_train,y_train)


# In[43]:


s.score(x_test,y_test)


# In[44]:


from sklearn.ensemble import RandomForestClassifier
f = RandomForestClassifier(criterion='entropy')
f.fit(x_train,y_train)


# In[45]:


f.score(x_train,y_train)


# In[46]:


f.score(x_test,y_test)


# In[47]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[48]:


from sklearn.svm import SVC
d = SVC(kernel='linear')
d.fit(x_train,y_train)


# In[49]:


d.score(x_train,y_train)


# In[50]:


d.score(x_test,y_test)


# In[51]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(x_train, y_train)


# In[52]:


classifier.score(x_train,y_train)


# In[53]:


classifier.score(x_test,y_test)


# In[ ]:




