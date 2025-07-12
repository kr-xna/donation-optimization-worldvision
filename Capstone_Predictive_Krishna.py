#!/usr/bin/env python
# coding: utf-8

# Load dataset, Downcasting and Sampling

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Function to load datasets with necessary columns
def load_dataset(filepath, usecols):
    return pd.read_csv(filepath, sep='\t', header=0, encoding='ISO-8859-1', usecols=usecols, low_memory=False)

# Load the necessary datasets with selected variables
responses = load_dataset('Deakin Bounceback Responses.txt', [
    'Deakin_SupporterID', 'TotalPaid', 'FulfilmentDate', 'ProductTemplateCode',
    'CatalogueItemDescription', 'Response_BPRChannelName', 'Response_InboundChannelName'
])

supporter_demographics = load_dataset('Deakin Bounceback Supporter_Demographics.txt', [
    'Deakin_SupporterID', 'Supporter_Type', 'DateOfBirth', 'Helix_Community', 'Helix_Persona'
])

contacted = load_dataset('Deakin Bounceback Contacted.txt', [
    'Deakin_SupporterID', 'Event_Date', 'Activity_Name'
])

audiences = load_dataset('Deakin Bounceback Audiences.txt', [
    'Deakin_SupporterID', 'NEW_TO_BB_FLAG', 'BBK_CAMPAIGN_SEGMENT'
])

# Apply 50% sampling to each dataset BEFORE merge
responses_sampled = responses.sample(frac=0.5, random_state=42)
supporter_demographics_sampled = supporter_demographics.sample(frac=0.5, random_state=42)
contacted_sampled = contacted.sample(frac=0.5, random_state=42)
audiences_sampled = audiences.sample(frac=0.5, random_state=42)

# Merge datasets on 'Deakin_SupporterID'
merged_df = pd.merge(responses_sampled, supporter_demographics_sampled, on='Deakin_SupporterID', how='left')
merged_df = pd.merge(merged_df, contacted_sampled, on='Deakin_SupporterID', how='left')
merged_df = pd.merge(merged_df, audiences_sampled, on='Deakin_SupporterID', how='left')

# Feature Engineering

# Convert 'DateOfBirth' to 'Age' if available
if 'DateOfBirth' in merged_df.columns:
    merged_df['DateOfBirth'] = pd.to_datetime(merged_df['DateOfBirth'], errors='coerce')
    merged_df['Age'] = 2024 - merged_df['DateOfBirth'].dt.year

# Convert 'FulfilmentDate' to datetime and extract the year for trend analysis
if 'FulfilmentDate' in merged_df.columns:
    merged_df['FulfilmentDate'] = pd.to_datetime(merged_df['FulfilmentDate'], errors='coerce')
    merged_df['Donation_Year'] = merged_df['FulfilmentDate'].dt.year

# Convert 'Event_Date' to datetime and extract the year
if 'Event_Date' in merged_df.columns:
    merged_df['Event_Date'] = pd.to_datetime(merged_df['Event_Date'], errors='coerce')
    merged_df['Event_Year'] = merged_df['Event_Date'].dt.year

# Drop the original datetime columns as they've been transformed into numerical features
merged_df = merged_df.drop(['DateOfBirth', 'FulfilmentDate', 'Event_Date'], axis=1, errors='ignore')

# Convert categorical columns to numeric using one-hot encoding
merged_df = pd.get_dummies(merged_df, drop_first=True)

# Handling Missing Values
# Drop rows with missing target variable 'TotalPaid'
merged_df = merged_df.dropna(subset=['TotalPaid'])

# Fill missing values for the other columns
for col in merged_df.select_dtypes(include=['float', 'int']).columns:
    merged_df[col].fillna(merged_df[col].median(), inplace=True)

for col in merged_df.select_dtypes(include=['object']).columns:
    merged_df[col].fillna(merged_df[col].mode()[0], inplace=True)

# Define the feature matrix X and the target variable y
X = merged_df.drop('TotalPaid', axis=1)
y = merged_df['TotalPaid']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building Predictive Models

# Option 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluate Linear Regression Model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

print("Linear Regression Model - Mean Squared Error:", mse_lr)
print("Linear Regression Model - R^2 Score:", r2_lr)
print("Linear Regression Model - Mean Absolute Error:", mae_lr)

# Option 2: Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Evaluate Decision Tree Model
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)

print("Decision Tree Model - Mean Squared Error:", mse_tree)
print("Decision Tree Model - R^2 Score:", r2_tree)
print("Decision Tree Model - Mean Absolute Error:", mae_tree)

# Option 3: Random Forest Regressor (Ensemble)
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Forest Model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("Random Forest Model - Mean Squared Error:", mse_rf)
print("Random Forest Model - R^2 Score:", r2_rf)
print("Random Forest Model - Mean Absolute Error:", mae_rf)

# Option 4: Gradient Boosting Regressor (Ensemble)
gb_model = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# Evaluate Gradient Boosting Model
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)

print("Gradient Boosting Model - Mean Squared Error:", mse_gb)
print("Gradient Boosting Model - R^2 Score:", r2_gb)
print("Gradient Boosting Model - Mean Absolute Error:", mae_gb)

# Visualizing the Predictions

plt.figure(figsize=(16, 10))

# Plot for Linear Regression
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_lr, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Linear Regression: Predicted vs Actual')
plt.xlabel('Actual TotalPaid')
plt.ylabel('Predicted TotalPaid')

# Plot for Decision Tree
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_tree, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Decision Tree: Predicted vs Actual')
plt.xlabel('Actual TotalPaid')
plt.ylabel('Predicted TotalPaid')

# Plot for Random Forest
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_rf, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Random Forest: Predicted vs Actual')
plt.xlabel('Actual TotalPaid')
plt.ylabel('Predicted TotalPaid')

# Plot for Gradient Boosting
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_gb, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Gradient Boosting: Predicted vs Actual')
plt.xlabel('Actual TotalPaid')
plt.ylabel('Predicted TotalPaid')

plt.tight_layout()
plt.show()

# Display model performance
print("\nModel Performance Summary:")
print(f"Linear Regression: MSE = {mse_lr:.2f}, R^2 = {r2_lr:.2f}, MAE = {mae_lr:.2f}")
print(f"Decision Tree: MSE = {mse_tree:.2f}, R^2 = {r2_tree:.2f}, MAE = {mae_tree:.2f}")
print(f"Random Forest: MSE = {mse_rf:.2f}, R^2 = {r2_rf:.2f}, MAE = {mae_rf:.2f}")
print(f"Gradient Boosting: MSE = {mse_gb:.2f}, R^2 = {r2_gb:.2f}, MAE = {mae_gb:.2f}")


# Initial Data Exploration

# In[ ]:


# Merge sampled datasets on 'Deakin_SupporterID'
merged_sampled_df = pd.merge(responses_sampled, supporter_demographics_sampled, on='Deakin_SupporterID', how='left')
merged_sampled_df = pd.merge(merged_sampled_df, contacted_sampled, on='Deakin_SupporterID', how='left')
merged_sampled_df = pd.merge(merged_sampled_df, portal_sampled, on='Deakin_SupporterID', how='left')
merged_sampled_df = pd.merge(merged_sampled_df, child_eletter_sampled, on='Deakin_SupporterID', how='left')
merged_sampled_df = pd.merge(merged_sampled_df, audiences_sampled, on=['Deakin_SupporterID', 'JobNumber'], how='left')

# Handle missing values
# Drop rows where 'TotalPaid' is missing (as it's our label/target variable)
merged_sampled_df = merged_sampled_df.dropna(subset=['TotalPaid'])

# For other missing values, we can fill them depending on the context:
# For numerical columns, we might fill with the median or mean
for col in merged_sampled_df.select_dtypes(include=['float', 'int']).columns:
    merged_sampled_df[col].fillna(merged_sampled_df[col].median(), inplace=True)

# For categorical columns, we can fill with the mode (most frequent value)
for col in merged_sampled_df.select_dtypes(include=['object']).columns:
    merged_sampled_df[col].fillna(merged_sampled_df[col].mode()[0], inplace=True)

import matplotlib.pyplot as plt

# Responses Dataset: Time series and Product distribution
responses_sampled['FulfilmentDate'] = pd.to_datetime(responses_sampled['FulfilmentDate'])
responses_sampled.groupby('FulfilmentDate')['TotalPaid'].sum().plot(kind='line', title='Total Donations Over Time')
plt.xlabel('Fulfilment Date')
plt.ylabel('Total Donations')
plt.show()

responses_sampled['ProductTemplateCode'].value_counts().plot(kind='bar', title='Distribution of Product Types')
plt.xlabel('Product Template Code')
plt.ylabel('Count')
plt.show()

# Supporter Demographics Dataset: Supporter Type and Helix Community distribution
supporter_demographics_sampled['Supporter_Type'].value_counts().plot(kind='bar', title='Distribution of Supporter Types')
plt.xlabel('Supporter Type')
plt.ylabel('Count')
plt.show()

supporter_demographics_sampled['Helix_Community'].value_counts().plot(kind='bar', title='Distribution of Helix Community')
plt.xlabel('Helix Community')
plt.ylabel('Count')
plt.show()

# Contacted Dataset: Interaction Category distribution and Time series analysis
contacted_sampled['InteractionCategory'].value_counts().plot(kind='bar', title='Distribution of Interaction Categories')
plt.xlabel('Interaction Category')
plt.ylabel('Count')
plt.show()

contacted_sampled['Event_Date'] = pd.to_datetime(contacted_sampled['Event_Date'])
contacted_sampled.groupby('Event_Date').size().plot(kind='line', title='Contact Events Over Time')
plt.xlabel('Event Date')
plt.ylabel('Number of Contacts')
plt.show()

# Portal Dataset: Login frequency analysis
portal_sampled['Registered_Datetime'] = pd.to_datetime(portal_sampled['Registered_Datetime'])
portal_sampled['LoggedIn_Datetime'] = pd.to_datetime(portal_sampled['LoggedIn_Datetime'])
portal_sampled['LoggedIn_Datetime'].hist(bins=50)
plt.title('Login Frequency Distribution')
plt.xlabel('Login Date')
plt.ylabel('Frequency')
plt.show()

# Child Eletter Dataset: eLetter interaction analysis
child_eletter_sampled['Sponsor_to_Child_eLetter'].value_counts().plot(kind='pie', title='Sponsor to Child eLetter Distribution', autopct='%1.1f%%')
plt.show()

child_eletter_sampled['Child_to_Sponsor_eLetter'].value_counts().plot(kind='pie', title='Child to Sponsor eLetter Distribution', autopct='%1.1f%%')
plt.show()

# Audiences Dataset: Distribution analysis for selected columns
print(audiences_sampled['NEW_TO_BB_FLAG'].value_counts())
print(audiences_sampled['PC_FLAG'].value_counts())
print(audiences_sampled['NON_RESPONDENT_FLAG'].value_counts())
print(audiences_sampled['CAMPAIGN_SINGLE_MULTI_SPONSOR'].value_counts())

audiences_sampled['DOLLAR_HANDLE_AMOUNT'].plot(kind='hist', title='Dollar Handle Amount Distribution')
plt.xlabel('Dollar Handle Amount')
plt.ylabel('Frequency')
plt.show()


# Correlation matrix

# In[ ]:


# Step 1: Convert date columns to datetime if not already done
date_columns = ['FulfilmentDate', 'Creation_DateTime', 'DateOfBirth', 'Event_Date',
                'Registered_Datetime', 'LoggedIn_Datetime', 'Sent_Date']
for col in date_columns:
    if col in merged_sampled_df.columns:
        merged_sampled_df[col] = pd.to_datetime(merged_sampled_df[col], errors='coerce')

# Step 2: Convert categorical columns to numeric where applicable
if 'NEW_TO_BB_FLAG' in merged_sampled_df.columns:
    merged_sampled_df['NEW_TO_BB_FLAG'] = merged_sampled_df['NEW_TO_BB_FLAG'].map({'Y': 1, 'N': 0})
if 'PC_FLAG' in merged_sampled_df.columns:
    merged_sampled_df['PC_FLAG'] = merged_sampled_df['PC_FLAG'].map({'Y': 1, 'N': 0})
if 'NON_RESPONDENT_FLAG' in merged_sampled_df.columns:
    merged_sampled_df['NON_RESPONDENT_FLAG'] = merged_sampled_df['NON_RESPONDENT_FLAG'].map({'Y': 1, 'N': 0})

# Step 3: Drop non-numeric columns that are not needed for the correlation matrix
numeric_df = merged_sampled_df.select_dtypes(include=['number']).copy()

# Step 4: Create the correlation matrix
corr_matrix = numeric_df.corr()

# Step 5: Visualize the correlation matrix using seaborn's heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix on Sampled Data')
plt.show()


# Feature Engineering

# In[ ]:


# Convert 'DateOfBirth' to 'Age' if the column exists
if 'DateOfBirth' in merged_sampled_df.columns:
    merged_sampled_df['Age'] = 2024 - merged_sampled_df['DateOfBirth'].dt.year

# Create 'Campaign_Length' as the difference between 'FulfilmentDate' and 'Event_Date'
if 'FulfilmentDate' in merged_sampled_df.columns and 'Event_Date' in merged_sampled_df.columns:
    merged_sampled_df['Campaign_Length'] = (merged_sampled_df['FulfilmentDate'] - merged_sampled_df['Event_Date']).dt.days

# Create 'Engagement_Duration' as the difference between 'LoggedIn_Datetime' and 'Registered_Datetime'
if 'Registered_Datetime' in merged_sampled_df.columns and 'LoggedIn_Datetime' in merged_sampled_df.columns:
    merged_sampled_df['Engagement_Duration'] = (merged_sampled_df['LoggedIn_Datetime'] - merged_sampled_df['Registered_Datetime']).dt.days

# Create 'Interaction_Frequency' as the count of contact events per supporter
interaction_freq = merged_sampled_df.groupby('Deakin_SupporterID').size().reset_index(name='Interaction_Frequency')
merged_sampled_df = pd.merge(merged_sampled_df, interaction_freq, on='Deakin_SupporterID', how='left')

# Drop columns that will not be used in the model
columns_to_drop = ['DMRT_Commitment_Key', 'Deakin_SupporterID', 'JobNumber', 'FulfilmentDate', 'Event_Date',
                   'Creation_DateTime', 'DateOfBirth', 'Registered_Datetime', 'LoggedIn_Datetime', 'Sent_Date']
merged_sampled_df = merged_sampled_df.drop(columns=columns_to_drop, errors='ignore')

# Convert any remaining categorical variables to numerical using one-hot encoding
merged_sampled_df = pd.get_dummies(merged_sampled_df, drop_first=True)

# Display the first few rows of the processed data
print(merged_sampled_df.head())

# Display the selected variables
print(merged_sampled_df.columns)


# Feature selection

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Define Features (X) and Target (y)
X = merged_sampled_df.drop('TotalPaid', axis=1)  # Features
y = merged_sampled_df['TotalPaid']  # Target variable

# Step 2: Remove low-variance features
selector = VarianceThreshold(threshold=0.01)  # Adjust the threshold as necessary
X_reduced = selector.fit_transform(X)

print("Shape after variance thresholding:", X_reduced.shape)

# Step 3: Train an initial RandomForest model to get feature importances
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_reduced, y)

# Get feature importances
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Keep only top 100 important features or a relevant number
X_important = X_reduced[:, indices[:100]]

print("Shape after selecting top 100 important features:", X_important.shape)

# Step 4: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_important, y, test_size=0.2, random_state=42)

# Step 5: Train a Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
y_pred_tree = tree_model.predict(X_test)

# Evaluate the model performance
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("Decision Tree Model - Mean Squared Error:", mse_tree)
print("Decision Tree Model - R^2 Score:", r2_tree)

# Step 6: Feature importance from the Decision Tree
feature_importance_tree = pd.Series(tree_model.feature_importances_, index=[X.columns[i] for i in indices[:100]])
feature_importance_tree = feature_importance_tree.sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance_tree.plot(kind='bar')
plt.title('Feature Importance from Decision Tree Model')
plt.show()


# Re-train the Model with Selected Features

# In[ ]:


# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_important, y, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Make predictions
y_pred_tree = tree_model.predict(X_test)

# Evaluate the model performance
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print("Decision Tree Model - Mean Squared Error:", mse_tree)
print("Decision Tree Model - R^2 Score:", r2_tree)

# Feature importance from the Decision Tree
feature_importance_tree = pd.Series(tree_model.feature_importances_, index=[X.columns[i] for i in indices[:100]])
feature_importance_tree = feature_importance_tree.sort_values(ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
feature_importance_tree.plot(kind='bar')
plt.title('Feature Importance from Decision Tree Model')
plt.show()


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select the final variables for analysis
final_df = merged_sampled_df[[
    'TotalPaid', 'DOLLAR_HANDLE_AMOUNT', 'NEW_TO_BB_FLAG', 'PC_FLAG',
    'NON_RESPONDENT_FLAG', 'CAMPAIGN_CHILD_COUNT',
    'Sponsor_to_Child_eLetter', 'Child_to_Sponsor_eLetter',
    'Activity_Code_x', 'Activity_Code_y'
]]

# One-hot encode categorical variables
final_df = pd.get_dummies(final_df, columns=['Activity_Code_x', 'Activity_Code_y'], drop_first=True)

# Feature Engineering: Interaction Term (e.g., NEW_TO_BB_FLAG and DOLLAR_HANDLE_AMOUNT)
final_df['NEW_TO_BB_DOLLAR'] = final_df['NEW_TO_BB_FLAG'] * final_df['DOLLAR_HANDLE_AMOUNT']

# Split the data into training and testing sets
X = final_df.drop('TotalPaid', axis=1)
y = final_df['TotalPaid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model: Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Evaluate the model
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Train R^2:", r2_score(y_train, y_pred_train))
print("Test R^2:", r2_score(y_test, y_pred_test))

# Display final selected features
print("Selected Features for Final Analysis:")
print(X.columns.tolist())


# In[ ]:


#To convert to pdf file.
get_ipython().system('apt-get install -y texlive-xetex texlive-fonts-recommended texlive-plain-generic')
get_ipython().system('apt-get install -y pandoc')

from google.colab import drive
drive.mount('/content/drive')

get_ipython().system("jupyter nbconvert --to pdf '/content/drive/MyDrive/Deakin Bounceback Data_Version 2/Capstone_Predictive.ipynb'")

