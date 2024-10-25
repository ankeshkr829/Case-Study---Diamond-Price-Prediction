#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Users\ankes\OneDrive\Desktop\Data\diamonds.csv")


# In[3]:


df.head()


# # Data Description

# Feature------------	Description
# 
# price	price in US dollars --($ 326 - $ 18,823)
# 
# carat	weight of the diamond --(0.2 - 5.01)
# 
# cut	quality of the cut --(Fair, Good, Very Good, Premium, Ideal)
# 
# color --diamond colour, (J (worst) to D (best))
# 
# clarity	a measurement of how clear the diamond is -- (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
# x	--length in mm (0 - 10.74)
# y	--width in mm (0 - 58.9)
# z	--depth in mm (0 - 31.8)
# 
# depth --total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43 - 79)
# 
# table --width of top of diamond relative to widest point (43 - 95)

# # Step 2 - Exploratory Data Analysis

# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


# Univariate - Categorical Analysis

fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
fig.suptitle("Univariate Plotting - Categorical Features")

axs[0].set_title("Cut")
sns.countplot(data=df, x='cut', ax=axs[0])

axs[1].set_title("Color")
sns.countplot(data=df, x='color', ax=axs[1])

axs[2].set_title("Clarity")
sns.countplot(data=df, x='clarity', ax=axs[2])


# In[8]:


# Bivariate - Categorical vs Numerical Analysis

fig, axs = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
fig.suptitle("Numerical vs Categorical Feature Visualization with Seaborn")

axs[0].set_title("Cut vs Price")
sns.boxplot(data=df, x='price', y='cut', ax=axs[0])

axs[1].set_title("Color vs Price")
sns.boxplot(data=df, x='price', y='color', ax=axs[1])

axs[2].set_title("Clarity vs Price")
sns.boxplot(data=df, x='price', y='clarity', ax=axs[2])

plt.show()


# In[9]:


# Univariate - Numerical Analysis

fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
fig.suptitle("Univariate Plotting - Numerical Features")

axs[0].set_title("Carat")
sns.histplot(data=df, x='carat', ax=axs[0])

axs[1].set_title("Depth")
sns.histplot(data=df, x='depth', ax=axs[1])

axs[2].set_title("Table")
sns.histplot(data=df, x='table', ax=axs[2])


# In[10]:


# Bivariate - Numerical vs Numerical Analysis

sns.pairplot(data=df,
           x_vars=['carat', 'depth', 'table'],
           y_vars='price')


# In[11]:


# Univariate - Numerical Analysis

fig, axs = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
fig.suptitle("Univariate Plotting - Numerical Features")

axs[0].set_title("X")
sns.histplot(data=df, x='x', ax=axs[0])

axs[1].set_title("Y")
sns.histplot(data=df, x='y', ax=axs[1])

axs[2].set_title("Z")
sns.histplot(data=df, x='z', ax=axs[2])


# In[12]:


# Bivariate - Numerical vs Numerical Analysis

sns.pairplot(data=df,
           x_vars=['x', 'y', 'z'],
           y_vars='price')


# # Data Preparation and Model Building Pipeline

# 1.Identifying the inputs (X) and output (y)
# 
# 2.Split into train and test (X_train, X_test, y_train, y_test)
# 3.Data Preparation: Data Cleaning and Feature Engineering.
#    * Clean the training data
#    * Preprocess the training data (X_train_transformed)
# 4.Training Phase: Build a model
# 5.Preprocess the test data (X_test_transformed)
# 6.Predict on unseen data
# 7.Evaluate the model performance

# In[13]:


import sklearn
print(sklearn.__version__)


# # Step 3 - Segregate Inputs (X) and Output (y)

# In[14]:


# Define the predictors (X) and target variable (y)

X = df.drop(columns=['price'])

y = df['price']


# # Step 4 - Split the data into Train and Test

# In[15]:


# Split into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = 0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# # Step 5 - Apply Data Preparation on Training Data

# In[16]:


X_train.dtypes


# Separate Numerical and Categorical Features

# In[17]:


# Separating Categorical and Numerical Columns

X_train_cat = X_train.select_dtypes(include=['object'])
X_train_num = X_train.select_dtypes(include=['int64', 'float64'])


# In[18]:


X_train_cat.head()


# In[19]:


X_train_num.head()


# In[20]:


# Rescaling numerical features
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()

# column names are (annoyingly) lost after Scaling
# (i.e. the dataframe is converted to a numpy ndarray)
X_train_num_transformed = pd.DataFrame(minmax_scaler.fit_transform(X_train_num), 
                                    columns = minmax_scaler.get_feature_names_out(), 
                                    index = X_train_num.index)

X_train_num_transformed.head()


# In[21]:


# Let's now analyse the properties of 'minmax_scaler'

print("Number of Numerical Features:", minmax_scaler.n_features_in_)
print("Output Feature Names:", minmax_scaler.get_feature_names_out())
print("Minimum of each column:", minmax_scaler.data_min_)
print("Maximum of each column:", minmax_scaler.data_max_)


# In[22]:


# Let's also describe the transformed data statistics

X_train_num_transformed.describe().round(2)


# In[23]:


# Rescaling numerical features
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

# column names are (annoyingly) lost after Scaling
# (i.e. the dataframe is converted to a numpy ndarray)
X_train_num_transformed = pd.DataFrame(std_scaler.fit_transform(X_train_num), 
                                    columns = std_scaler.get_feature_names_out(), 
                                    index = X_train_num.index)

X_train_num_transformed.head()


# In[24]:


# Let's now analyse the properties of 'std_scaler'

print("Number of Numerical Features:", std_scaler.n_features_in_)
print("Output Feature Names:", std_scaler.get_feature_names_out())
print("Mean of each column:", std_scaler.mean_)
print("Std of each column:", np.sqrt(std_scaler.var_))


# In[25]:


# Let's also describe the transformed data statistics

X_train_num_transformed.describe().round(2)


# # Categorical Feature Transformation: Applying One-Hot Encoding (Note: We won't use OHE Transformation)

# In[26]:


# OneHotEncoding the categorical features
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse_output=False, 
                               handle_unknown="ignore")

# column names are (annoyingly) lost after OneHotEncoding
# (i.e. the dataframe is converted to a numpy ndarray)
X_train_cat_tansformed = pd.DataFrame(onehot_encoder.fit_transform(X_train_cat), 
                               columns=onehot_encoder.get_feature_names_out(), 
                               index = X_train_cat.index)

print("Shape of Data before Transformation:", X_train_cat.shape)
print("Shape of Data after Transformation:", X_train_cat_tansformed.shape)


# In[27]:


# Let's now analyse the properties of 'onehot_encoder'

print("Applied encoding on:", onehot_encoder.feature_names_in_)
print("Unique Categories:", onehot_encoder.categories_)
print("Feature Names after encoding:", onehot_encoder.get_feature_names_out())


# Categorical Feature Transformation: Applying One-Hot Encoding with drop='first'

# In[28]:


# OneHotEncoding the categorical features
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(drop='first', 
                               sparse_output=False, 
                               handle_unknown="ignore")

# column names are (annoyingly) lost after OneHotEncoding
# (i.e. the dataframe is converted to a numpy ndarray)
X_train_cat_tansformed = pd.DataFrame(onehot_encoder.fit_transform(X_train_cat), 
                               columns=onehot_encoder.get_feature_names_out(), 
                               index = X_train_cat.index)

print("Shape of Data before Transformation:", X_train_cat.shape)
print("Shape of Data after Transformation:", X_train_cat_tansformed.shape)

X_train_cat_tansformed.head()


# In[29]:


# Let's now analyse the properties of 'onehot_encoder'

print("Applied encoding on:", onehot_encoder.feature_names_in_)
print("Unique Categories:", onehot_encoder.categories_)
print("Feature Names after encoding:", onehot_encoder.get_feature_names_out())


# Categorical Feature Transformation: Applying One-Hot Encoding with drop='first' and min_frequency=3000 (Note: We won't use OHE Transformation)

# In[30]:


# OneHotEncoding the categorical features
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(drop='first', 
                               min_frequency=3000, 
                               sparse_output=False, 
                               handle_unknown="ignore")

# column names are (annoyingly) lost after OneHotEncoding
# (i.e. the dataframe is converted to a numpy ndarray)
X_train_cat_tansformed = pd.DataFrame(onehot_encoder.fit_transform(X_train_cat), 
                               columns=onehot_encoder.get_feature_names_out(), 
                               index = X_train_cat.index)

print("Shape of Data before Transformation:", X_train_cat.shape)
print("Shape of Data after Transformation:", X_train_cat_tansformed.shape)

X_train_cat_tansformed.head()


# In[31]:


# Let's now analyse the properties of 'onehot_encoder'

print("Applied encoding on:", onehot_encoder.feature_names_in_)
print("Unique Categories:", onehot_encoder.categories_)
print("Infrequent Categories", onehot_encoder.infrequent_categories_)
print("Feature Names after encoding:", onehot_encoder.get_feature_names_out())


# # Categorical Feature Transformation: Applying Label Encoding (Note: We won't use Label Encoding Transformation)

# Encode target labels with value between 0 and n_classes-1.
As per documentation, this transformer should be used to encode target values, i.e. y, and not the input X
# In[32]:


example_df = pd.DataFrame({'Rating': ['Excellent', 'Average', 'Bad', 'Average', 'Excellent', 'Bad', 'Good', 'Good']})

example_df


# In[33]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

example_df['Ratings - Label Encoded'] = label_encoder.fit_transform(example_df['Rating'])

example_df


# # Categorical Feature Transformation: Applying Ordinal Encoding

# As the three categorical features are ordinal, we will proceed with the Ordinal Encoding instead of One-Hot Encoding or Label Encoding

# In[34]:


# Define the ordering for categorical columns (lowest to highest)

cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_categories = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']


# In[35]:


# Create the OrdinalEncoder with the specified categories
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories],  
                                 handle_unknown="use_encoded_value",
                                 unknown_value=-1, 
                                 encoded_missing_value=-5)

# Apply the encoding
X_train_cat_transformed = pd.DataFrame(ordinal_encoder.fit_transform(X_train_cat), 
                                     columns=ordinal_encoder.get_feature_names_out(), 
                                     index = X_train_cat.index)

X_train_cat_transformed.head()


# In[36]:


# Let's now analyse the properties of 'ordinal_encoder'

print("Applied encoding on:", ordinal_encoder.feature_names_in_)
print("Unique Categories:", ordinal_encoder.categories_)
print("Feature Names after encoding:", ordinal_encoder.get_feature_names_out())


# # Concatenate X_train_num_transformed and X_train_cat_transformed

# In[37]:


X_train_transformed = pd.concat([X_train_num_transformed, X_train_cat_transformed], axis=1)

X_train_transformed.head()


# # Step 7- Apply Data Preparation on Test Data

# Note that, Step-6 is discussed after this.

# In[38]:


# Separate Categorical and Numerical Features

X_test_cat = X_test.select_dtypes(include=['object'])
X_test_num = X_test.select_dtypes(include=['int64', 'float64'])


# In[39]:


# Apply transformation on Numerical data

X_test_num_transformed = pd.DataFrame(std_scaler.transform(X_test_num), 
                                   columns = std_scaler.get_feature_names_out(), 
                                   index = X_test_num.index)

X_test_num_transformed.head()


# In[40]:


# Apply transformation on Categorical data

X_test_cat_transformed = pd.DataFrame(ordinal_encoder.transform(X_test_cat), 
                                   columns = ordinal_encoder.get_feature_names_out(), 
                                   index = X_test_cat.index)

X_test_cat_transformed.head()


# In[41]:


# Concatinate X_test_num_transformed and X_test_cat_transformed

X_test_transformed = pd.concat([X_test_num_transformed, X_test_cat_transformed], axis=1)

X_test_transformed.head()


# # Step 6, 8 and 9 - Training and Testing Phase (Linear Regression)

# In[42]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn import metrics\n\nregressor = LinearRegression()\nregressor.fit(X_train_transformed, y_train)\n\ny_test_pred = regressor.predict(X_test_transformed)\n\nprint("Model\'s Error:", metrics.mean_absolute_error(y_test, y_test_pred))\nprint()\n')


# In[43]:


output_df = pd.DataFrame({'Actual': y_test})


# In[44]:


output_df['Linear Regression Predictions'] = y_test_pred

output_df


# In[45]:


fig, ax = plt.subplots(figsize=(8,3))

sns.histplot(output_df['Actual'], color='blue', alpha=0.5, label="actual")
sns.histplot(output_df['Linear Regression Predictions'], color='red', alpha=0.5, label="prediction")

plt.legend()


# # Step 6, 8 and 9 - Training and Testing Phase (KNN Regression)

# In[46]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.neighbors import KNeighborsRegressor\nfrom sklearn import metrics\n\nregressor = KNeighborsRegressor()\nregressor.fit(X_train_transformed, y_train)\n\ny_test_pred = regressor.predict(X_test_transformed)\n\nprint("Model\'s Error:", metrics.mean_absolute_error(y_test, y_test_pred))\nprint()\n')


# In[47]:


output_df['KNN Regression Predictions'] = y_test_pred

output_df


# In[48]:


fig, ax = plt.subplots(figsize=(8,3))

sns.histplot(output_df['Actual'], color='blue', alpha=0.5, label="actual")
sns.histplot(output_df['KNN Regression Predictions'], color='red', alpha=0.5, label="prediction")

plt.legend()


# # Step 6, 8 and 9 - Training and Testing Phase (DT Regression)

# In[49]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn import metrics\n\nregressor = DecisionTreeRegressor()\nregressor.fit(X_train_transformed, y_train)\n\ny_test_pred = regressor.predict(X_test_transformed)\n\nprint("Model\'s Error:", metrics.mean_absolute_error(y_test, y_test_pred))\nprint()\n')


# In[50]:


output_df['DT Regression Predictions'] = y_test_pred

output_df


# In[51]:


fig, ax = plt.subplots(figsize=(8,3))

sns.histplot(output_df['Actual'], color='blue', alpha=0.5, label="actual")
sns.histplot(output_df['DT Regression Predictions'], color='red', alpha=0.5, label="prediction")

plt.legend()


# # Step 6, 8 and 9 - Training and Testing Phase (Random Forest Regression)

# In[52]:


get_ipython().run_cell_magic('time', '', '\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn import metrics\n\nregressor = RandomForestRegressor()\nregressor.fit(X_train_transformed, y_train)\n\ny_test_pred = regressor.predict(X_test_transformed)\n\nprint("Model\'s Error:", metrics.mean_absolute_error(y_test, y_test_pred))\nprint()\n')


# In[53]:


output_df['RF Regression Predictions'] = y_test_pred

output_df


# In[54]:


fig, ax = plt.subplots(figsize=(8,3))

sns.histplot(output_df['Actual'], color='blue', alpha=0.5, label="actual")
sns.histplot(output_df['RF Regression Predictions'], color='red', alpha=0.5, label="prediction")

plt.legend()


# # Comparing all the Models

# In[55]:


df_melted = pd.melt(output_df,  var_name='Model', value_name='Prediction')

df_melted.head()


# In[56]:


fig, ax = plt.subplots(figsize=(8,3))

sns.boxplot(x='Prediction', y='Model', data=df_melted)
plt.title('Comparison of Actual vs. Predicted Values')

plt.show()


# In[ ]:




