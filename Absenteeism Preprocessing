# In[1]:
import pandas as pd #pandas = panel data
# In[2]:

raw_csv_data = pd.read_csv("C:/Users///Desktop/Data Science Bootcamp/Absenteeism_data.csv")

# In[3]:
raw_csv_data
# In[4]:

df = raw_csv_data.copy()

# In[5]:
df
# In[6]:

pd.options.display.max_columns = None
pd.options.display.max_rows = None
display(df)

# In[7]:

df.info() #check to see if there are any missing values in the columns 
### Task: Predict absenteeism from work
# In[8]:
# Drop ID
#ID is a label variable- a number that is there to distinguish the individuals from one another, not to carry any numeric information. (nominal data)

df.drop(['ID'], axis=1)
# In[9]:

df = df.drop(['ID'],axis=1) # we have to equal it to itself to make the change. And the change is permanent

# In[10]:

df
### 'Reason for absence'
# In[11]:
df['Reason for Absence'].min()

# In[12]:
df['Reason for Absence'].max()

# In[13]:
pd.unique(df['Reason for Absence']) #Which reasons for absence were used?

# In[14]:
df['Reason for Absence'].unique()

# In[15]:
len(df['Reason for Absence'].unique()) #0-28 means there are 29 different reasons. We have 28. Which one is missing?

# In[16]:
sorted(df['Reason for Absence'].unique()) #We are missing number 20

### .get_dummies()
# In[17]:
reason_columns = pd.get_dummies(df['Reason for Absence'],dtype=int)

# In[18]:
reason_columns

# In[19]:

reason_columns['check'] = reason_columns.sum(axis=1)
reason_columns

# In[20]:

reason_columns['check'].sum(axis=0)

# In[21]:
reason_columns['check'].unique()

# In[22]:

reason_columns = reason_columns.drop(['check'],axis=1)
reason_columns

### Drop the 'Reason 0' to avoid multicollinearity

# In[23]:
#reason_columns.drop(reason_columns[[0]], axis=1) <- I tried this method before watching the video but the video uses the method below
reasons_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True, dtype='int')
reasons_columns

### Group the Reasons for Absence:
# In[24]:
df.columns.values #This will show you the names of the columns

# In[25]:
reasons_columns.columns.values

### Step 1 Drop the Reason for Absence column
# In[26]:
df = df.drop(['Reason for Absence'], axis=1)
df

### Group the columns of Reasons_columns
# In[27]:
# Grouping these variables = Classification
# Re-organizing a certain type of variables into groups in a regression analysis
# Group  = Class

# In[28]:


reason_columns.loc[:,22:]

# In[29]:
#Here we are grouping the reasons into 4 classes. What we can do here is group the classes even more. We do this by checking if
#There is a reason used within that group. We know the maximum amount of reasons used in all observations is 1. Because of this,
#We can use the max() function to group them and see if one the reasons was used from within that group from all 700 observations. 

reason_type_1 = reasons_columns.loc[:,1:14].max(axis=1)
reason_type_2 = reasons_columns.loc[:,15:17].max(axis=1)
reason_type_3 = reasons_columns.loc[:,18:21].max(axis=1)
reason_type_4 = reasons_columns.loc[:,22:].max(axis=1)

# In[30]:


reason_type_2

### Concatenate the Column Values

# In[31]:
df

# In[32]:
df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
df


# In[33]:
# Adding Column Names to the Reason type classes

df.columns.values

# In[34]:
column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']


# In[35]:
df.columns = column_names
df.head()

### Reorder Columns
# In[36]:
column_names_reordered = [ 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']


# In[37]:
df = df[column_names_reordered]
df.head()

### Create a Checkpoint

# In[38]:
# Create a copy of the current state of the df dataframe
df_reason_mod = df.copy()
df_reason_mod.head()

### Date variable
# In[39]:

type(df_reason_mod['Date'][0])  #This code gives you the type from the first element of the column 'Date'

# In[40]:
#Timestamp a classical data type found in many programming languages out there, used for values representing dates and time
# pd.to_datetime() converts values into timestamp

df_reason_mod['Date'] = pd.to_datetime(df_reason_mod['Date'], format = '%d/%m/%Y')
df_reason_mod.head()


# In[41]:
type(df_reason_mod['Date'][0])
### Extract the Month Value

# In[42]:

df_reason_mod['Date'][0]


# In[43]:

df_reason_mod['Date'][0].month # The index 0 is giving you the first element of the 'Date' column


# In[44]:

list_months = []
list_months


# In[45]:
#for i in range(700):
    #list_months.append(df_reason_mod['Date'][i].month)
#list_months
#This is a good method but this is a better method 

for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)  
    # .append() attaches the new calue obtained from each iteration to the existing content of the desginated list


# In[46]:
len(list_months)


# In[47]:
df_reason_mod['Month Value'] = list_months
df_reason_mod.head()

### Extract the Day of the Week:
# In[48]:
# Lets see what day the element 699 (700th observation) is 

df_reason_mod['Date'][699].weekday() # This will give you '3'. Days are numbers 0-6. 0 =  Monday, 6 = Friday.


# In[49]:
df_reason_mod['Date'][699] # May 31st of 2018 was a Thursday. (3=Thursday)

### Lets create a function for this

# In[50]:


def date_to_weekday(date_value):
    return date_value.weekday()


# In[51]:


df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)
df_reason_mod.head()


# In[52]:

df_reason_mod = df_reason_mod.drop(['Date'], axis=1)
df_reason_mod.head()


# In[53]:
df_reason_mod.columns


# In[54]:


reordered_columns_mo_day = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Month Value', 'Day of the Week',
       'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
       'Pets', 'Absenteeism Time in Hours']


# In[55]:


df_reason_mod = df_reason_mod[reordered_columns_mo_day]
df_reason_mod.head()


# In[56]:


df_reason_date_mod = df_reason_mod.copy()
df_reason_date_mod.head()


# In[57]:


type(df_reason_date_mod['Transportation Expense'][0])


# In[58]:


type(df_reason_date_mod['Distance to Work'][0])


# In[59]:


type(df_reason_date_mod['Age'][0])


# In[60]:


type(df_reason_date_mod['Daily Work Load Average'][0])


# In[61]:


type(df_reason_date_mod['Body Mass Index'][0])


# In[62]:
# Children and Pets tells you the number of children or pets they have so we dont have to change those values.
# Education does have another meaning so we have to look more into that variable


# In[63]:
df_reason_date_mod['Education'] # 1-High school 2-graduate 3-Postgraduate 4-Master or PhD


# In[64]:
df_reason_date_mod['Education'].value_counts()  #This tells you how many times each value comes out on the dataset


# In[65]:
# There are almost 600 High School and barely over 100 of Degree holders. Lets just combine the degree holders


# In[66]:
#Instead of using the dummy function, use the .map() function and create a dictionary


# In[67]:

df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
df_reason_date_mod['Education'].unique()


# In[73]:
df_reason_date_mod['Education'].value_counts()


# In[75]:

display(df_reason_date_mod)

### Final Checkpoint

# In[76]:


df_preprocessed = df_reason_date_mod.copy()
df_preprocessed.head(10)

### Export to a csv file on the machine

# In[86]:


df_preprocessed.to_csv("C:/Users/camay/OneDrive/Desktop/Data Science Bootcamp/df_preprocessed.csv", index=False)





