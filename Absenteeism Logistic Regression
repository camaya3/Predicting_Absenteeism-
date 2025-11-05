## Stating the obvious but this was done using Juptyer
# ## Creating a logistic regression to predict absentism
# ### Import the relevant libraries

# In[1]:

import pandas as pd
import numpy as np

### Load the data
# In[2]:

data_preprocessed = pd.read_csv("C:/Users/camay/OneDrive/Desktop/Data Science Bootcamp/df_preprocessed.csv")
data_preprocessed.head()
# ### Create the targets

# In[3]:
# Were going to use the median to measure if someone has been excessive or not in terms of absenteeism

# In[4]:

data_preprocessed['Absenteeism Time in Hours'].median() #cutoff is going to be 3


# In[5]:


# If the observation value is less than 3 we will assign it the value of 0. Otherwise, 1. These are the targets


# In[6]:


targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)
targets


# In[7]:


data_preprocessed['Excessive Absenteeism'] = targets
data_preprocessed.head()

### A comment on the targets

# In[8]:
#By using the median and not the value '3' we are implicitly balancing the dataset. Roughly half of the targets are 0's and other half 1's

# In[9]:
targets.sum()/targets.shape[0] #Around 46% are 1's. You want 50/50 but sometimes a 60/40 will work. 45/55 will work better. 

# In[10]:

data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours','Day of the Week',
                                            'Daily Work Load Average','Distance to Work'], axis=1)
data_with_targets.head()

# In[11]:
#Check to see if it is a checkpoint and we have changed the dataframe
# In[12]:
data_with_targets is data_preprocessed

# ### Select the inputs
# In[13]:

data_with_targets.shape

# In[14]:

data_with_targets.iloc[:,:14]

# In[15]:

data_with_targets.iloc[:,:-1]

# In[16]:

unscaled_inputs = data_with_targets.iloc[:,:-1]

### Standardize the data

# In[17]:

from sklearn.preprocessing import StandardScaler

absenteeism_scaler  = StandardScaler()


# In[18]:
# We standardized the dummy's and we were not supposed to. We have to select the inputs that we want to standardize. This is how.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns


    def fit(self,X):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

# In[19]:


unscaled_inputs.columns.values


# In[20]:

columns_to_scale=['Month Value','Day of the Week', 'Transportation Expense', 'Distance to Work',
       'Age', 'Daily Work Load Average', 'Body Mass Index','Children', 'Pets']

columns_to_omit = ['Reason_1','Reason_2','Reason_3','Reason_4','Education']

# In[21]:

# List comprehension is a syntatic construct which allows us to create list from exisitng list absed on loops, conditionals, etc. 
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]

# In[22]:
absenteeism_scaler = CustomScaler(columns_to_scale)
# In[23]:
absenteeism_scaler.fit(unscaled_inputs)
# In[24]:
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)
scaled_inputs

# In[25]:
scaled_inputs.shape
### Split the data into train and test and shuffle
#### Import the relevant module

# In[26]:
from sklearn.model_selection import train_test_split

#### Split the data
# In[27]:
train_test_split(scaled_inputs,targets)

# In[28]:

x_train, x_test, y_train, y_test = train_test_split(scaled_inputs,targets, train_size = .8, random_state = 20)
#The default split is 25/75 so adjust train_size for 10/90. Train_test_split also shuffles randomly but that can be a problem. 
#Instead set the random_state to an integer so that the method always shuffles the observations the same 'random' way. 

# In[29]:

print(x_train.shape, y_train.shape)
# In[30]:
print(x_test.shape, y_test.shape)

### Logistic Regression with Sklearn
# In[31]:
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics

#### Training the model
# In[32]:
reg = LogisticRegression()
# In[33]:
reg.fit(x_train,y_train)
# In[34]:
reg.score(x_train,y_train) #Based on the data we used, our model learned to classify ~80% (.78392) of the observations correctly

### Manually check the accuracy 
# In[35]:
model_outputs = reg.predict(x_train)
model_outputs

# In[36]:
y_train

# In[37]:
model_outputs == y_train

# In[38]:

np.sum(model_outputs == y_train)

# In[39]:

print(np.sum(model_outputs == y_train)/model_outputs.shape[0]) #This is what the .score() method gave us. We're good. 

### Finding the intercepts and the coefficients
# In[40]:
# We want to find the formula so that we can get the intercept (b0) and coefficients (x1,x2,x3,x4,...)
#This is because that is the formula we can use on other software like Tableau and SQL and it describes out data for future use
# In[41]:

reg.intercept_
# In[42]:
reg.coef_
# In[43]:

feature_name = unscaled_inputs.columns.values
feature_name

# In[44]:

summary_table = pd.DataFrame (columns=['Feature Name'], data=feature_name)
summary_table['Coefficient'] = np.transpose(reg.coef_)
summary_table

# In[45]:

summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table

# In[46]:

summary_table['Odds Ratio'] = np.exp(summary_table.Coefficient)
summary_table

# In[47]:
# Sort the dataframe using the new column Odds Ratio

summary_table.sort_values('Odds Ratio', ascending=False)
###### Interpretation: For a unit change in the standardized feature, the odds increase by a mulitple equal to the odds ratio (1 = no change)
### Testing the model
# In[48]:

reg.score(x_test,y_test)

# In[49]:
predicted_proba = reg.predict_proba(x_test)
predicted_proba


# In[51]:

predicted_proba[:,1]
### Save the model

# In[52]:
import pickle

# In[53]:
with open('model','wb') as file:   # open('file name','wright bytes' )   when you unpickle you'll use rb or right bytes
    pickle.dump(reg,file)          # dump means save 
# In[54]:

with open('scaler','wb') as file:
    pickle.dump(absenteeism_scaler,file)


#### Clever approach (Creating a module)

# In[55]:


#Instead of feeding new data into the script and having to change certain syntax to make it work, create a module


# In[ ]:


#Check course file

