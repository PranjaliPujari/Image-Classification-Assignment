#!/usr/bin/env python
# coding: utf-8

# # HOSPITAL RATING CLASSIFICATION

# Welcome to the Starter Code for the Hospital Rating Classification Capstone Project!
# 
# In this notebook you'll find 
# - A blueprint on how to attempt the course project.
# - Additional hints and directions on different tasks
# 
# Please note that this approach is one of the many approaches you can take for solving this Capstone project.

# ### Import the necessary libraries

# In[1]:


import pandas as pd, numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from pandas.core.common import random_state

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score
# Import 'MinMaxScaler' from 'sklearn'
from sklearn.preprocessing import MinMaxScaler


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# ### Task 1 
# 
# Task 1: Understand the data
# 
#  
# 
# Take some time to familiarize yourself with the data. What are the key variables?
# 
# Specifically, answer the following questions:
# 
# - 1.1 - Perform a few basic data quality checks to understand the different columns and prepare descriptive statistics for some of the important columns.
# - 1.2 - What is the distribution of hospital overall ratings? How are they varying across other parameters like `State`? Create  a few visualizations that provide some insights into the data.

# ##### Task 1.1

# In[3]:


df = pd.read_csv('hospital-info.csv')      ## Write the code to load the dataset 


# In[4]:


df.head()


# In[5]:


##Check the data type of the different columns
## Hint - You can use the .info() method here
df.info()


# Note down your observations after completing the above task. You should ask questions such as:
# 
# - Check for non-null columns. Do you see any column having missing values?
# - Are the datatypes correct for all the variables? You might have to convert a few of them to categorical later

# > There are no missing or null value in given dataset.
# 
# > Yes, some variables have incorrect datatype. I will convert a few of them to categorical when it needed.

# ##### Task 1.2

# In[6]:


## Descriptive Statistics
## Let's take the main measures and the hospital overall rating first.
features = ['Mortality national comparison', 'Safety of care national comparison',
            'Readmission national comparison', 'Patient experience national comparison',
            'Effectiveness of care national comparison', 'Timeliness of care national comparison',
            'Efficient use of medical imaging national comparison']


# In[7]:


#making dataframe of main measures
new_df = df[['Hospital overall rating','Mortality national comparison',                                 
         'Safety of care national comparison','Readmission national comparison',\
         'Patient experience national comparison','Effectiveness of care national comparison',\
         'Timeliness of care national comparison','Efficient use of medical imaging national comparison']].copy() 


# In[8]:


### Filter out the above columns from the DataFrame and compute the descriptive statistics
### Hint - The .describe() method might be useful
filtered_df=df.drop(labels=None, axis=0, index=None, columns=features, level=None, inplace=False, errors='raise')


# In[9]:


filtered_df.describe()       #descriptive statistics


# In[10]:


### Analyze your target variable - "Hospital overall rating"
### How does the ratings distribution look like for all the hospitals?

## Hint - You can use seaborn plots like countplot() for checking distribution of the ratings
## Hint - You can plot a correlation heatmap to check the correlation between the main measures stored in "features"
## Hint - You can also check the correlations between the "Hospital overall rating" and other measures


# In[11]:


# countplot of hospital overall rating
plt.figure(figsize = (13, 6))
sns.countplot(x = 'Hospital overall rating', data = filtered_df)
xt = plt.xticks(rotation=45)
plt.xlabel('Hospital overall rating')
plt.ylabel('Count')
plt.title('Hospital overall rating')                   


# In[12]:


import pandas as pd
new_df2 = df[['Hospital overall rating','Mortality national comparison', 'Safety of care national comparison',
            'Readmission national comparison', 'Patient experience national comparison',
            'Effectiveness of care national comparison', 'Timeliness of care national comparison',
            'Efficient use of medical imaging national comparison']].copy()
new_df2 = new_df2.astype(str)
print(new_df2.dtypes)                       #converting the datatypes of some variables 


# In[13]:


features = ['Hospital overall rating','Mortality national comparison',         'Safety of care national comparison','Readmission national comparison',         'Patient experience national comparison','Effectiveness of care national comparison',         'Timeliness of care national comparison','Efficient use of medical imaging national comparison']

for i, item in enumerate(features):
    plt.figure(figsize = (13, 6))
    sns.countplot(x = item, data = new_df2)
    xt = plt.xticks(rotation=45)
    plt.xlabel(item)
    plt.ylabel('Count')
    plt.title(item)                         #visualisation of main measures in countplot


# In[14]:


sns.heatmap(df[features].corr(),annot=True)              # a correlation heatmap to check the correlation between the main measures stored in "features"


# In[15]:


#correlations between the "Hospital overall rating" and other measures
corr = new_df.corr()
corr


# In[16]:


### Check how the hospital ratings vary across other parameters
### Hint - Some example parameters are "State" and "Hospital Ownership"
### Hint - You can use the pivot_table functionality of pandas to perform this


# In[17]:


#checkig how the hospital rating vary across state parameter and hospital ownership parameter.
pivot = df.pivot_table(index=['Hospital overall rating', 'State', 'Hospital Ownership'], aggfunc='mean')
pivot


# 
# Note down your observations after completing the above task. You should ask questions such as:
# 
# - How are ratings distributed? the Are you seeing any peculiar distributions for the ratings?
# - How do the correlations between the measures and the target variable look like?
# - How do ratings vary across the different levels of the parameter that you have taken?

# > Hospital rating is given from 1 to 5. Maximum 3 rating is observed and minimum 1 rating is observed among all the hospitals.
# 
# > Mostly all variables have positive co relation with target variable. 
# 
# > Ratings are distributed among all types of hospitals. There is no specific category observed in low rating or in high rating.

# ### Task 2 - Building machine learning models
# 
# Use your knowledge of classification models to create three models that predict hospital ratings. You should follow these steps:
# 
# - Prepare the data for the machine learning model 
#    - Remove all the demographic columns as well as any other uneccessary features from the data set
#    - For simplification, instead of having 5 ratings, we will convert them to 0 and 1. Here 0 indicates that the hospital has been rated 3 or below and 1 indicates that the hospital has been rated as 4 or 5.  Encode the Hospital columns as follows
#             1,2,3 : 0
#             4,5: 1
#    - Store the predictors and the target variable in variables X and y.
#    - Create the dummy variables for categorical columns.
#    - Split the data into train and test sets (70-30 split with random state 0. This random state is recommended, though you can use any other random state of your choice).
#    - Scale the numerical columns using StandardScaler.
# - Build 3 classification models on your dataset. Carefully apply regularization and hyperparameter tuning techniques to improve your model performance for each of the models.
# - Summarize the classification performance in terms of the necessary metrics such as accuracy, sensitivity, specificity, etc.

# #####  Prepare the data for machine learning model

# In[18]:


## Drop all the demographic features
demo_features = ['Provider ID','Hospital Name',
 'Address',
 'City',
 'State',
 'ZIP Code',
 'County Name',
 'Phone Number']


# In[19]:


## Drop all the above features from the DataFrame df and store the rest of the features in df2
df2 = df.drop(labels=None, axis=0, index=None, columns=demo_features, level=None, inplace=False, errors='raise')


# In[20]:


### Check the first 5 rows of df2 to see if the drop operation has worked correctly or not
df2.head()


# In[21]:


##Recheck the columns to see if anything else needs to be dropped
## There might be other unnecessary columns that require dropping


# > There is no any unnecessary columns that required dropping.

# ##### Map the ratings 
# 
# - 1,2,3 will be 0
# - 4,5 will be 1

# In[22]:


## Hint -  Write a simple lambda function to do the mapping
## Refer to this link from Course 1 for more help -  https://learn.upgrad.com/course/2897/segment/16179/128948/394776/2054363


# In[23]:


new_df1 = df2[['Hospital overall rating']].copy()       # making a new subset  of hospital overall rating


# In[24]:


#using lambda function to convert the rating in o or 1.
df4=new_df1[['Hospital overall rating']].apply(lambda x: 0 if x['Hospital overall rating'] == 1  else 0 if x['Hospital overall rating'] == 2 else 0 if x['Hospital overall rating'] == 3 else 1, axis=1)
df6 = pd.concat([new_df1,df4], axis=1)
df6.rename(columns = {'o':'value'}, inplace = True)
df6


# ##### Convert the datatypes of the categorical variables

# In[25]:


### In task 1, you would have identified the categorical variables, which may or may not be in their correct data types
### Now is the right time to convert them to the correct datatype 
### This will be useful when you create dummy variables next


# In[26]:


new_df2 = df[['Hospital overall rating','Mortality national comparison', 'Safety of care national comparison',
            'Readmission national comparison', 'Patient experience national comparison',
            'Effectiveness of care national comparison', 'Timeliness of care national comparison',
            'Efficient use of medical imaging national comparison']].copy()
new_df2 = new_df2.astype(str)
print(new_df2.dtypes)                       #converting the datatypes


# ##### Data Preparation and Train-test split

# In[27]:


### Create X and y variable
X = df.drop(columns=['Hospital overall rating'])
y = df['Hospital overall rating']


# In[28]:


### Create the dummy variables for categorical variables
### Note - Make sure the "drop_first parameter" is correctly initialized for different ML models
### Hint - You can create multiple versions of the X dataset
X = pd.get_dummies(X, drop_first=False) #for kNN and trees
X2 = pd.get_dummies(X, drop_first=True) #for regression


# In[29]:


## Perform the train_test split to create the train and validation sets
## Choose any random state of your choice 
## Split it in the ratio of 70-30
X_train, X_val, X2_train, X2_val, y_train, y_val = train_test_split(X, X2, y, test_size=0.3, random_state = 1)


# In[30]:


# Scale and Standardize the numerical variables
scaler = StandardScaler()
numeric_cols = [col for col in X.columns if X[col].dtypes != 'object']

X_train[numeric_cols]= scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols]= scaler.transform(X_val[numeric_cols])


# In[31]:


X_scaled = scaler.fit_transform(X)
X2_train, X2_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state = 1)


# #####  Model building and evaluation
# 
# You have learned multiple classification models till now, such as logistic regression, k-NN and decision trees. You should choose 3 of the models for performing the tasks in this section. You can follow the below steps:
# 
# 
# - Build the models and fit it on training data
# - Perform classifications on the validation data
# - Compute and tabulate the validation accuracies for the different models
# - Compare the accuracies for the different models and choose the best model
# 
# **Note** - You can also evaluate your models using additional metrics like `F1 score`, `Sensitivity`,`Specificity` , etc.
# 
# 
# **Helpful Resource** - For writing precise code for this section, you can refer to the code you learned in Model Selection Lab Session in the `kNN and Model Selection` module.
# 
# 
# 

# - Additional notes
#   - You can peform additional tasks like building ROC/AUC curves for all the models and identifying an optimal cut-off
#   - You can also build conjectures around some arbitrary metric cut-offs. For example, say you want to build a model which has atleast 50% accuracy, specificity and sensitivity. Use these conjectures to arrive at a final model
#   - Note that there is no right answer for this particular question. You will be awarded marks as long as your overall approach is correct

# # Simple linear model

# In[32]:


#building a regression model
regression_model = LinearRegression()
regression_model.fit(X2_train, y_train)

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[idx]))


# **Here the coeficient values are relatively smaller. So we can say this is the smoother model.**

# In[33]:


intercept = regression_model.intercept_

print("The intercept for our model is {}".format(intercept))


# # Regularized Ridge Model 

# In[34]:


ridge = Ridge(alpha=.3) #coefficients are prevented to become too big by this alpha value
ridge.fit(X2_train,y_train)
for i,col in enumerate(X_train.columns):   
    print ("Ridge model coefficients for {} is {}:".format(col,ridge.coef_[i]))


# **We can see less coefficients values compared to linear regression. Since it is not a smoother model we will see  difference in coefficient.**

# # Regularized LASSO Model

# In[35]:


lasso = Lasso(alpha=0.1)
lasso.fit(X2_train,y_train)
for i,col in enumerate(X2_train):
    print ("Lasso model coefficients for {} is {}:".format(col,lasso.coef_[i]))


# **Many of the coefficients have become 0 so we can drop of those dimensions from the model.It has taken only 5 dimensions to build the model.Lasso is also used for feature selection.**

# ## Comparing the scores

# In[36]:


print(regression_model.score(X2_train, y_train))
print(regression_model.score(X2_val, y_val))


# In[37]:


print(ridge.score(X2_train, y_train))
print(ridge.score(X2_val, y_val))


# **Accuracy of linear and ridge are more or less same because both coefficients values are similar**

# In[38]:


print(lasso.score(X2_train, y_train))
print(lasso.score(X2_val, y_val))


# In[39]:


# Linear model - USE LassoCV to get the best LASSO model

from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV

lasso_model = Lasso(alpha=1)
lasso_model.fit(X2_train, y_train)


# In[40]:


#rmse function 
def rmse(y_train, y_pred):
  return np.sqrt(mean_squared_error(y_train, y_pred))


# In[41]:


print('RMSE training set', round(rmse(y_train, lasso_model.predict(X2_train)), 1))
print('RMSE validation set', round(rmse(y_val, lasso_model.predict(X2_val)), 1))


# In[42]:


coef_table = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(lasso_model.coef_))], axis = 1)
coef_table.columns = ['', 'Coefficient']
coef_table.set_index('', inplace=True)
coef_table


# In[43]:


alphas = np.arange(0.01, 50, .5) # Check all alphas from .01 to 10 in steps of size 0.25
lasso_cv_model = LassoCV(alphas= alphas , cv=5, max_iter=50000)
# fit model
lasso_cv_model.fit(X2_train, y_train)
# summarize chosen configuration
print('alpha: %f' % lasso_cv_model.alpha_)


# In[44]:


# Set best alpha
lasso_best_model = Lasso(alpha=lasso_cv_model.alpha_)
lasso_best_model.fit(X2_train, y_train)

print('Best Lasso-RMSE training set', round(rmse(y_train, lasso_best_model.predict(X2_train)), 3))
print('Best Lasso-RMSE validation set', round(rmse(y_val, lasso_best_model.predict(X2_val)), 3))


# In[45]:


# Tree Model - Use max depth to control the complexity of the tree. Run a Grid search for multiple values of max depth.

tree_model_1 = DecisionTreeClassifier(random_state = 1)
tree_model_1 = tree_model_1.fit(X_train, y_train)


# In[46]:


# Visualize the decision tree for 'tree_model_1'
# Note: This cell may take a while to run owing to the large number of nodes to be displayed, so please be patient

fig = plt.figure(figsize = (16,8))
fig = tree.plot_tree(tree_model_1, feature_names = X.columns, filled = True)


# In[47]:


# Print the number of leaves and the depth of the tree for 'tree_model_1' using the 'get_n_leaves()' and the 'get_depth()' methods
tree_model_1_n_leaves = tree_model_1.get_n_leaves()
tree_model_1_depth = tree_model_1.get_depth()
print('Number of leaves =', tree_model_1_n_leaves)
print('Tree depth =', tree_model_1_depth)


# In[48]:


# Obtain predicted class labels for the training and validation data using 'tree_model_1' using the 'predict()' method
# Hint: Study the documentation of the 'predict()' method
y_pred_1_train = tree_model_1.predict(X_train)
y_pred_1_val = tree_model_1.predict(X_val)


# In[49]:


# Compute the accuracy and the sensitivity of 'tree_model_1' on the training and validation data
# Note: The 'pos_label' parameter for the 'recall_score()' method should be set to the positive class label
# Note: The positive class label in this exercise is '1', which is also the default value for the 'pos_label' parameter
acc_train_1 = accuracy_score(y_train, y_pred_1_train)
acc_val_1 = accuracy_score(y_val, y_pred_1_val)

# Summarize the above metrics for the train and validation sets using a single data frame and display it
tree_model_1_metrics = pd.DataFrame(data = {'Accuracy': [acc_train_1, acc_val_1],
                                            },
                                    index = ['tree_model_1_train', 'tree_model_1_val'])

tree_model_1_metrics


# In[50]:


y = df['Hospital overall rating']
y_train, y_val = train_test_split(y, test_size=0.3, random_state = 1)


# In[51]:


#rmse function 
def rmse(y_train, y_pred):
  return np.sqrt(mean_squared_error(y_train, y_pred))


# In[52]:


# KNN Model

##### CODE HERE #####
X_train= scaler.fit_transform(X_train)
X_val= scaler.transform(X_val)



# Find the value of k for which RMSE is minimum, using GridSearchCV

##### CODE HERE #####
kvalues = np.arange(1,31) # Parameter range

val_rmse=[]

for k in kvalues:
  knn_reg = KNeighborsRegressor(n_neighbors=k)
  knn_reg.fit(X_train, y_train)
  y_pred = knn_reg.predict(X_val)
  val_rmse.append(rmse(y_val, y_pred))

# Find the value of k for which RMSE is minimum, using GridSearchCV

##### CODE HERE #####
### Rmse vs k
plt.plot(kvalues,val_rmse,marker='o')
plt.xlabel("k")
plt.ylabel("rmse")
print("The minimum rmse is obtained at k = " + str(np.argmin(val_rmse)+1))


# In[53]:


knn_reg_best = KNeighborsClassifier(n_neighbors=29)
knn_reg_best.fit(X_train, y_train)
print('RMSE validation set:', round(rmse(y_val, knn_reg_best.predict(X_val)), 2))


# In[54]:


##Let's fit the best model and calculate some classification performance measures


# In[55]:


knn_clf_best = KNeighborsClassifier(n_neighbors=22)
knn_clf_best.fit(X_train, y_train)
y_pred = knn_clf_best.predict(X_val)


# In[56]:


plt.rcParams.update({'font.size': 14}) # To make the plot labels easier to read
ConfusionMatrixDisplay.from_estimator(
        knn_clf_best,
        X_val,
        y_val,
        cmap=plt.cm.Blues,
    )


# In[57]:


# Check all alphas from .01 to 10 in steps of size 0.25
alphas = np.arange(.01, 25, .25)
lasso_cv_model = LassoCV(alphas= alphas, cv=5, max_iter=50000)
lasso_cv_model.fit(X2_train, y_train)

# Train the lasso regression model with the best value of alpha
lin_reg_best = Lasso(alpha=lasso_cv_model.alpha_)
lin_reg_best.fit(X2_train, y_train)

# Calculate RMSE for the lasso regression model on train and validation sets
lin_train_rmse = rmse(y_train, lin_reg_best.predict(X2_train))
lin_val_rmse = rmse(y_val, lin_reg_best.predict(X2_val))


# In[58]:


# Get the different values of ccp alphas
tree_reg = DecisionTreeRegressor()
path= tree_reg.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas = path.ccp_alphas

# Train decision tree regressor models for different values of ccp_alpha

# Create a list to store the different tree models
regs=[]

# Iterate through different ccp alpha values and train models for each of them
for ccp_alpha in ccp_alphas:
    # Create and train the model
    curr_reg = DecisionTreeRegressor(random_state=0, ccp_alpha = ccp_alpha)
    curr_reg.fit(X2_train,y_train)

# Save the model in the list
    regs.append(curr_reg)

# Calculate the RMSE for all the tree models

# Create lists to store RMSE on training and validation data sets
    train_rmse=[]
    val_rmse=[]

# Iterate through the models and calculate RMSE
for r in regs:
    y_train_pred=r.predict(X2_train)
    y_val_pred = r.predict(X2_val)

    train_rmse.append(rmse(y_train_pred,y_train))
    val_rmse.append(rmse(y_val_pred,y_val))

# Pick the best ccp alpha
    best_ccp_alpha = ccp_alphas[val_rmse.index(min(val_rmse))]

# Train the corresponding tree
    tree_reg_best = tree.DecisionTreeRegressor(random_state=0, ccp_alpha=best_ccp_alpha)
    tree_reg_best.fit(X_train,y_train)

# Calculate RMSE for the best tree
    tree_train_rmse = rmse(y_train, tree_reg_best.predict(X_train))
    tree_val_rmse = rmse(y_val, tree_reg_best.predict(X_val))


# In[59]:


# KNN Model (using our previous knowledge about the best k!)
knn_reg_best = KNeighborsRegressor(n_neighbors=16)
knn_reg_best.fit(X_train, y_train)

knn_train_rmse = rmse(y_train, knn_reg_best.predict(X_train))
knn_val_rmse = rmse(y_val, knn_reg_best.predict(X_val))


# In[60]:


# Create a dataframe to display the RMSE values for the three models
pd.DataFrame([[lin_train_rmse, lin_val_rmse],[tree_train_rmse, tree_val_rmse], 
              [knn_train_rmse, knn_val_rmse]], columns=['RMSE Train', 'RMSE Validation'], 
              index = ['Linear', 'Tree', 'kNN'])


# In[61]:


# Perform train-validation split on the new independent variable
y_train, y_val = train_test_split(y, test_size=0.3, random_state = 1)


# In[62]:


# Create and train the logistic regression model
log_clf_best = LogisticRegression(penalty='none', solver='lbfgs', random_state=0, 
                                     max_iter=200).fit(X2_train, y_train)

# Calculate the logistic regression model's accuracy on the training and validation sets
log_train_acc = log_clf_best.score(X2_train, y_train)
log_val_acc = log_clf_best.score(X2_val, y_val)

# Create and train the decision tree classifier model
best_ccp_alpha = 0.004801587301587302 # from Module 4
tree_clf_best = DecisionTreeClassifier(random_state=0, ccp_alpha=best_ccp_alpha)
tree_clf_best.fit(X_train,y_train)

# Calculate the decision tree classifier model's accuracy on the training and validation sets
tree_train_acc = tree_clf_best.score(X_train, y_train)
tree_val_acc = tree_clf_best.score(X_val, y_val)

# Create and train the kNN model
knn_clf_best = KNeighborsClassifier(n_neighbors=14)
knn_clf_best.fit(X_train, y_train)

# Calculate the kNN model's accuracy on the training and validation sets
knn_train_acc = knn_clf_best.score(X_train, y_train)
knn_val_acc = knn_clf_best.score(X_val, y_val)

# Create a dataframe to display the accuracy values for the three models
pd.DataFrame([[log_train_acc, log_val_acc], [tree_train_acc, tree_val_acc], 
              [knn_train_acc, knn_val_acc]], columns=['Training Acc', 'Validation Acc'], 
              index = ['Logistic', 'Tree', 'kNN'])


# In[63]:


# Set potential cutoff values
clf_cutoffs = np.arange(0,1.01,0.01)

# Recover profit for training and testing data
profit_arr = np.array(df['Hospital overall rating'])
profit_train, profit_val = train_test_split(profit_arr, test_size=0.3, random_state = 1)

# For each model obtain the predicted probabilities and then save the total profit for different cutoffs
table = []
for clf_model in [log_clf_best, tree_clf_best, knn_clf_best]:
    
    # In case of logistic regression, use X2
    if clf_model == log_clf_best:
      probs_train = clf_model.predict_proba(X2_train)[:, 1]
      probs_val = clf_model.predict_proba(X2_val)[:, 1]
    
    # In case of decision tree or kNN, use X
    else:
      probs_train = clf_model.predict_proba(X_train)[:, 1]
      probs_val = clf_model.predict_proba(X_val)[:, 1]
    
    # Add the profits for all models which satisfy the cutoff for different values of cutoff
    clf_profits = []
    for cutoff in clf_cutoffs:
        clf_profits.append(sum(profit_train[probs_train > cutoff]))

    # Calculating the best cutoff
    best_profit_train = max(clf_profits)
    best_cutoff = clf_cutoffs[clf_profits.index(best_profit_train)]
    best_profit_val = sum(profit_val[probs_val > best_cutoff])
    table.append([best_cutoff, best_profit_train, best_profit_val])


# In[64]:


# Set potential cutoff values
reg_cutoffs = np.arange(-5000, 2000)

# For each model obtain the predicted profit and then save the total profit for different cutoffs
for reg_model in [lin_reg_best, tree_reg_best, knn_reg_best]:
    
    # In case of linear regression, use X2
    if reg_model == lin_reg_best:
      pred_train = reg_model.predict(X2_train) 
      pred_val = reg_model.predict(X2_val)
    
    # In case of decision tree or kNN, use X
    else:
      pred_train = reg_model.predict(X_train) 
      pred_val = reg_model.predict(X_val)

    # Add the profits for all models which satisfy the cutoff for different values of cutoff     
    reg_profits = []
    for cutoff in reg_cutoffs:
        reg_profits.append(sum(profit_train[pred_train > cutoff]))

    # Calculating the best cutoff
    best_profit_train = max(reg_profits)
    best_cutoff = reg_cutoffs[reg_profits.index(best_profit_train)]
    best_profit_val = sum(profit_val[pred_val > best_cutoff])
    table.append([best_cutoff, best_profit_train, best_profit_val])


# In[65]:


table_df = pd.DataFrame(table, columns=['Cutoff Value', 'Train Profit', 'Validation Profit'], 
                        index=['Logistic Clf','Tree Clf','KNN Clf','Linear Reg',' Tree Reg', 'KNN Reg'])
table_df


# ### Task 3 
# 
# You have now built (at least) three machine learning models. Choose the best model according to your metrics and provide the following recommendations
# -  Hospital Rating Predictor: Using the best model of your choice, predict the ratings of a few new hospitals which are yet to be assigned a rating by CMS. The information for these hospitals has been provided in a separate CSV file named 'not_yet_rated.csv'.
# - Hospital Improvement Plan: Let's say a few of the hospitals were rated low (0) by the model that you chose. Provide recommendations on how these hospitals can improve their ratings

# In[66]:


###Let's read the not_yet_rated dataset
new = pd.read_csv('not_yet_rated.csv')  


# In[67]:


## Check the top 5 rows
new.head()


# ### Approach to predict ratings
# - Perform the exact same data preparation steps as earlier
#    - Drop the unnecessary columns
#    - Convert the datatypes of categorical variables and create dummies
#    - Standardize the numeric columns
# - After that we shall use the `.predict()` method of your ML model to predict the ratings

# In[68]:


new.info()                  #checking null values and data type


# > There is no null value in given data set.
# > Some variables has incorrect data type. We will convert them when needed.

# In[69]:


import pandas as pd
new_2 = new[[ 'Safety of care national comparison','Readmission national comparison',
            'Patient experience national comparison','Effectiveness of care national comparison',
            'Timeliness of care national comparison',
            'Efficient use of medical imaging national comparison']].copy()
new_2 = new_2.astype(str)
print(new_2.dtypes)           # convert the datatype into object


# In[70]:


### Create X and y variable
X = new.drop(columns=['Provider ID']) 
y = new['Provider ID']


# In[71]:


### Create the dummy variables for categorical variables
### Note - Make sure the "drop_first parameter" is correctly initialized for different ML models
### Hint - You can create multiple versions of the X dataset
X = pd.get_dummies(X, drop_first=False) #for kNN and trees
X2 = pd.get_dummies(X, drop_first=True) #for regression


# In[72]:


## Perform the train_test split to create the train and validation sets
## Choose any random state of your choice 
## Split it in the ratio of 70-30
X_train, X_val, X2_train, X2_val, y_train, y_val = train_test_split(X, X2, y, test_size=0.3, random_state = 1)


# In[73]:


# Scale and Standardize the numerical variables
scaler = StandardScaler()
numeric_cols1 = [col for col in X.columns if X[col].dtypes != 'object']

X_train[numeric_cols1]= scaler.fit_transform(X_train[numeric_cols1])
X_val[numeric_cols1]= scaler.transform(X_val[numeric_cols1])


# In[74]:


X_scaled = scaler.fit_transform(X)
X2_train, X2_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state = 1)


# In[75]:


# Linear model - USE LassoCV to get the best LASSO model

from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV

lasso_model = Lasso(alpha=1)
lasso_model.fit(X2_train, y_train)


# In[76]:


#rmse function 
def rmse(y_train, y_pred):
  return np.sqrt(mean_squared_error(y_train, y_pred))


# In[77]:


print('RMSE training set', round(rmse(y_train, lasso_model.predict(X2_train)), 1))
print('RMSE validation set', round(rmse(y_val, lasso_model.predict(X2_val)), 1))


# In[78]:


coef_table = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(lasso_model.coef_))], axis = 1)
coef_table.columns = ['', 'Coefficient']
coef_table.set_index('', inplace=True)
coef_table.head(10)


# In[79]:


alphas = np.arange(0.01, 50, .5) # Check all alphas from .01 to 10 in steps of size 0.25
lasso_cv_model = LassoCV(alphas= alphas , cv=3, max_iter=50000)
# fit model
lasso_cv_model.fit(X2_train, y_train)
# summarize chosen configuration
print('alpha: %f' % lasso_cv_model.alpha_)


# In[80]:


# Set best alpha
lasso_best_model = Lasso(alpha=lasso_cv_model.alpha_)
lasso_best_model.fit(X2_train, y_train)

print('Best Lasso-RMSE training set', round(rmse(y_train, lasso_best_model.predict(X2_train)), 3))
print('Best Lasso-RMSE validation set', round(rmse(y_val, lasso_best_model.predict(X2_val)), 3))


# In[81]:


# Tree Model - Use max depth to control the complexity of the tree. Run a Grid search for multiple values of max depth.

tree_model_2= DecisionTreeClassifier(random_state = 1)
tree_model_2 = tree_model_2.fit(X_train, y_train)


# In[82]:


# Visualize the decision tree for 'tree_model_1'
# Note: This cell may take a while to run owing to the large number of nodes to be displayed, so please be patient

fig = plt.figure(figsize = (16,8))
fig = tree.plot_tree(tree_model_2, feature_names = X.columns, filled = True)


# In[83]:


# Print the number of leaves and the depth of the tree for 'tree_model_1' using the 'get_n_leaves()' and the 'get_depth()' methods
tree_model_2_n_leaves = tree_model_2.get_n_leaves()
tree_model_2_depth = tree_model_2.get_depth()
print('Number of leaves =', tree_model_2_n_leaves)
print('Tree depth =', tree_model_2_depth)


# In[84]:


# Obtain predicted class labels for the training and validation data using 'tree_model_1' using the 'predict()' method
# Hint: Study the documentation of the 'predict()' method
y_pred_1_train = tree_model_2.predict(X_train)
y_pred_1_val = tree_model_2.predict(X_val)


# In[85]:


# Compute the accuracy and the sensitivity of 'tree_model_1' on the training and validation data
# Note: The 'pos_label' parameter for the 'recall_score()' method should be set to the positive class label
# Note: The positive class label in this exercise is '1', which is also the default value for the 'pos_label' parameter
acc_train_1 = accuracy_score(y_train, y_pred_1_train)
acc_val_1 = accuracy_score(y_val, y_pred_1_val)

# Summarize the above metrics for the train and validation sets using a single data frame and display it
tree_model_1_metrics = pd.DataFrame(data = {'Accuracy': [acc_train_1, acc_val_1],
                                            },
                                    index = ['tree_model_1_train', 'tree_model_1_val'])

tree_model_1_metrics


# In[86]:


from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
# generate 2d classification dataset
X, y = make_blobs(n_samples=3057, centers=2, n_features=2, random_state=1)
# fit final model
model = LogisticRegression()
model.fit(X, y)
# new instances where we do not know the answer
Xnew, _ = make_blobs(n_samples=6, centers=2, n_features=2, random_state=1)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))          #prediction of rating


# ### Approach to identify areas of improvement
# 
# - Identify the measures which have a positive influence on the overall hospital ratings. For example,
#     - if you're using a logistic regression model, you can check the coefficients
#         - A +ve coefficient indicates +ve influence on the overall hospital rating
#         - A -ve coefficient indicates -ve influence on the overall hospital rating
# - Identify in which of the above measures a low-rated hospital is currently lagging behind. These measures need to be improved.
# - Further deep dive into the sub-measures using the same approach as above. 

# > With the help of logistic regression model, we can say that there is mostly positive influence of measures on the overall hospital rating.
# 
# > Safety of care national comparison, Efficient use of medical imaging national comparison, Timeliness of care national comparison measures need to be improved. Because of these measures a low rated hospital is currently lagging behind.
# 
# > Many measures have more or less influence on rating. There is need of some improvemnt.
