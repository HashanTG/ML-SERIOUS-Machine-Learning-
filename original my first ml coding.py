#!/usr/bin/env python
# coding: utf-8

# # Import initial libraries of pandas

# In[1]:


import pandas as pd


# # Read data set

# In[3]:


data=pd.read_csv('add.csv')


# In[4]:


data


# # Display head of the dataset

# In[5]:


data.head()


# # Display tail of the dataset

# In[6]:


data.tail()


# # Identity of the number of rows and coloumns of the data set

# In[7]:


data.shape


# # Preprocessing

# In[8]:


data.info()


# # Do the EDA

# In[9]:


import matplotlib.pyplot as plt


# # Identity pattern and decide the algorithm

# In[10]:


plt.scatter(data['x'],data['sum'])


# In[12]:


plt.scatter(data['y'],data['sum'])


# # Store feature matrix in X and pesponse(target) in  vecter y
# 

# In[13]:


X=data[['x','y']]
y=data['sum']


# # Train /test split

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)


# ### importance of random state after train

# In[16]:


X_train


# # Import and train the model

# In[17]:


from sklearn.linear_model import LinearRegression


# ### Creating the instance of the model

# In[19]:


model=LinearRegression()


# ### train the model

# In[20]:


model.fit(X_train,y_train)


# # Model's pediction performence checking(over fiting)

# In[21]:


model.score(X_train,y_train)


# In[22]:


model.score(X_test,y_test)


# # Compare the results

# In[23]:


y_pred=model.predict(X_test)


# In[24]:


y_pred


# In[25]:


df=pd.DataFrame({'Actual':y_test,'prediction':y_pred})


# In[26]:


df


# # Prediction for new samples(train and watch it true another 2 numbers)

# In[27]:


model.predict([[27,112]])


# # Now our dataset is train for another datas 

# # We want to save this model reuse with our developing(with pythin) interface

# # Saving process

# ### import joblib library

# In[28]:


import joblib


# ### saving our trained model as model_joblib

# In[30]:


joblib.dump(model,'model_joblib')


# ### we can load our trained library and get predictions(first load our saved model)

# In[31]:


model=joblib.load('model_joblib')


# ### now give two numbers and you can predict with our trained model

# In[32]:


model.predict([[23,45]])


# # Training this model for the entire data set (all data)

# ### asign data to X and y

# In[33]:


X=data[['x','y']]
y=data['sum']


# ### train etire dataset again with LinearRegression

# In[34]:


model=LinearRegression()
model.fit(X,y)


# # Save again this trained model as 'model_joblib'

# In[36]:


import joblib
joblib.dump(model,'model_joblib')


# ### recheck this saved model is working well 

# #### first load finally saved model 

# In[37]:


model=joblib.load('model_joblib')


# #### now you can check this model working propally

# In[38]:


model.predict([[23,45]])


# # GUI MAKING

# ### first import tkinter and make interface

# In[49]:


from tkinter import *
import joblib
master=Tk()
master.title("Addition of two number using ML")
label= Label(master,text="addition of two number using ML",bg='black',fg='white').grid(row=0,columnspan=2)


# ### put the function top of it and applying labels

# In[50]:


def show_entry_fields():
    p1=float(e1.get())
    p2=float(e2.get())
    
    
    model=joblib.load('model_joblib')
    result=model.predict([[p1,p2]])
    
    Label(master,text='sum is=').grid(row=4)
    Label(master,text=result).grid(row=5)
    
   

Label(master,text="Enter First Number").grid(row=1)
Label(master,text="Enter Second Number").grid(row=2)

e1=Entry(master)
e2=Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)

Button(master,text='predict',command=show_entry_fields).grid()

mainloop()


# In[ ]:




